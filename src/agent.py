"""
Deep Q-Network (DQN) Agent with Experience Replay
==================================================

Architecture
------------
  Input  : state vector (WINDOW_SIZE × n_features flattened)
  Hidden : 3 × Linear(256) + LayerNorm + ReLU + Dropout(0.1)
  Output : Q-values for each action (HOLD / BUY / SELL)

Self-learning mechanism
-----------------------
1. **Experience Replay** – past (s, a, r, s', done) tuples are stored in a
   circular buffer and sampled randomly, breaking temporal correlations.
2. **Target Network** – a periodically-synced copy of the online network
   provides stable Q-value targets, avoiding the "moving-target" problem.
3. **ε-greedy Exploration** – starts with high exploration (ε=1) and decays
   toward exploitation (ε→0.05), balancing discovery vs. known strategies.
4. **Prioritised mini-batch** – the loss gradient flows back to the online
   network each batch, so every mistake immediately shifts future decisions.
"""

import os
import logging
import random
from collections import deque
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """Dueling DQN with Layer Normalization for stable training."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Dueling streams: value V(s) and advantage A(s,a)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.feature(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────

Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size circular experience-replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent that learns from XAUUSD trading mistakes.

    Key self-learning components
    ----------------------------
    - Each bad trade → negative reward → lower Q-value for that action in
      that state → agent avoids repeating the mistake.
    - Each good trade → positive reward → higher Q-value → agent repeats it.
    - Target network prevents unstable feedback loops.
    - ε-decay forces the agent to commit to learned behaviour over time.
    """

    def __init__(self,
                 state_dim:  int   = config.STATE_DIM,
                 action_dim: int   = config.ACTION_DIM,
                 hidden_dim: int   = config.HIDDEN_DIM,
                 lr:         float = config.LEARNING_RATE,
                 gamma:      float = config.GAMMA,
                 epsilon:    float = config.EPSILON_START,
                 epsilon_end:float = config.EPSILON_END,
                 epsilon_decay: float = config.EPSILON_DECAY,
                 buffer_size: int  = config.REPLAY_BUFFER_SIZE,
                 batch_size:  int  = config.BATCH_SIZE):

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DQNAgent using device: %s", self.device)

        self.online_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self.train_steps = 0
        self.episode     = 0

    # ── Decision ─────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    # ── Memory ───────────────────────────────────────────────────────────────

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ── Learning from mistakes ────────────────────────────────────────────────

    def learn(self) -> float:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the scalar loss (for logging).

        Learning from mistakes, step by step:
        1. Sample random past experiences from the replay buffer.
        2. Compute target Q-values using the (stable) target network.
        3. Negative-reward transitions → target Q < current Q → network
           learns to predict low value for that (state, action) pair.
        4. Gradient descent updates weights so future predictions are better.
        """
        if len(self.buffer) < config.MIN_REPLAY_SIZE:
            return 0.0

        batch       = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values for the taken actions
        current_q = self.online_net(states_t).gather(1, actions_t)

        # Double DQN: online net picks action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states_t).gather(1, next_actions)
            target_q     = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def update_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)
        self.episode += 1

    def update_target_network(self) -> None:
        """Hard copy online → target (called every TARGET_UPDATE_FREQ episodes)."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target network updated at episode %d", self.episode)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "episode":     self.episode,
            "train_steps": self.train_steps,
        }, path)
        logger.info("Model saved → %s", path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon     = ckpt.get("epsilon",     self.epsilon_end)
        self.episode     = ckpt.get("episode",     0)
        self.train_steps = ckpt.get("train_steps", 0)
        logger.info("Model loaded ← %s  (episode=%d, ε=%.4f)",
                    path, self.episode, self.epsilon)
