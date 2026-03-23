"""
CNN-LSTM Double DQN Agent with Prioritized Experience Replay
=============================================================

Improvements over the basic DQN agent:

1. **Double DQN** — the online network selects the greedy next action;
   the target network evaluates it.  This prevents Q-value overestimation
   that causes the agent to be over-confident about mediocre strategies.

2. **Prioritized Experience Replay (PER)** — trades with large |TD error|
   (unexpected outcomes) are replayed more often, so the network learns
   faster from surprising wins/losses rather than repeating easy cases.

3. **CNN-LSTM-Dueling architecture** (HybridDQN) — learns its own features
   from raw OHLCV instead of hand-picked indicators.

4. **Hard target-network sync** every TARGET_SYNC_STEPS gradient steps,
   keeping the learning target stable while the online net trains quickly.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model_cnn_lstm import HybridDQN
from src.per_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

MODEL_TYPE_KEY = "v2_cnn_lstm"          # written into checkpoint for auto-detection


class CNNLSTMAgent:
    """
    Double DQN + PER agent backed by the HybridDQN network.

    The public interface mirrors DQNAgent so training/backtest code
    can swap them with minimal changes.
    """

    def __init__(
        self,
        action_dim:        int,
        hidden_dim:        int   = 256,
        lr:                float = 1e-4,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.05,
        epsilon_decay:     float = 0.9980,
        buffer_capacity:   int   = 100_000,
        batch_size:        int   = 64,
        min_replay:        int   = 2_000,
        per_alpha:         float = 0.6,
        per_beta_start:    float = 0.4,
        per_beta_steps:    int   = 300_000,
        target_sync_steps: int   = 500,     # gradient steps between hard syncs
    ):
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size   = batch_size
        self.min_replay   = min_replay
        self._target_sync = target_sync_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("CNNLSTMAgent using device: %s", self.device)

        # Networks
        self.online_net = HybridDQN(action_dim, hidden_dim).to(self.device)
        self.target_net = HybridDQN(action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr,
                                    weight_decay=1e-5)

        # PER buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity   = buffer_capacity,
            alpha      = per_alpha,
            beta_start = per_beta_start,
            beta_steps = per_beta_steps,
        )

        # Counters (persisted in checkpoint)
        self.episode     = 0
        self.train_steps = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        return self.online_net.obs_dim

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        self.online_net.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q = self.online_net(x)
        self.online_net.train()
        return int(q.argmax().item())

    # ── Memory ────────────────────────────────────────────────────────────────

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    # ── Learning ──────────────────────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.min_replay:
            return None

        states, actions, rewards, next_states, dones, weights, indices = \
            self.buffer.sample(self.batch_size)

        s  = torch.from_numpy(states).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        a  = torch.from_numpy(actions).long().to(self.device)
        r  = torch.from_numpy(rewards).float().to(self.device)
        d  = torch.from_numpy(dones).float().to(self.device)
        w  = torch.from_numpy(weights).float().to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_a    = self.online_net(ns).argmax(dim=1)           # online selects
            next_q    = self.target_net(ns).gather(                 # target evaluates
                            1, next_a.unsqueeze(1)).squeeze(1)
            target_q  = r + self.gamma * next_q * (1.0 - d)

        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Per-sample Huber loss, weighted by IS
        td_errors = (target_q - current_q).detach().cpu().numpy()
        elem_loss = nn.functional.huber_loss(current_q, target_q, reduction="none")
        loss      = (w * elem_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in PER buffer
        self.buffer.update_priorities(indices, td_errors)

        self.train_steps += 1

        # Periodic hard sync of target network
        if self.train_steps % self._target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug("Target network synced at step %d", self.train_steps)

        return float(loss.item())

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode += 1

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_type":    MODEL_TYPE_KEY,
                "online_net":    self.online_net.state_dict(),
                "target_net":    self.target_net.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
                "per":           self.buffer.state_dict(),
                "epsilon":       self.epsilon,
                "episode":       self.episode,
                "train_steps":   self.train_steps,
            },
            path,
        )
        logger.info("CNNLSTMAgent saved → %s  (ep=%d, ε=%.4f)",
                    path, self.episode, self.epsilon)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "per" in ckpt:
            self.buffer.load_state_dict(ckpt["per"])
        self.epsilon     = ckpt.get("epsilon",     self.epsilon_end)
        self.episode     = ckpt.get("episode",     0)
        self.train_steps = ckpt.get("train_steps", 0)
        logger.info("CNNLSTMAgent loaded ← %s  (ep=%d, ε=%.4f)",
                    path, self.episode, self.epsilon)
