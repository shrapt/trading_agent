"""
Prioritized Experience Replay (PER)
=====================================
Reference: Schaul et al. (2015) "Prioritized Experience Replay"
           https://arxiv.org/abs/1511.05952

Data structure: binary SumTree for O(log N) weighted sampling.

Key ideas:
  - Experiences that surprised the agent (large |TD error|) are replayed more
    often, so the network learns faster from unexpected outcomes.
  - Importance-sampling (IS) weights correct the bias introduced by non-uniform
    sampling so that the gradient update remains unbiased in expectation.
  - β starts at 0.4 and anneals to 1.0, giving full IS correction by the end
    of training when the policy is nearly converged.
"""

from __future__ import annotations
import numpy as np


# ── SumTree ───────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary tree whose leaf values are priorities and internal nodes hold
    the sum of their children.  Supports O(log N) add, update, and
    weighted random sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = [None] * capacity
        self._write   = 0           # next leaf to overwrite
        self.n_stored = 0           # how many valid entries exist

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        """Descend the tree to find the leaf whose cumulative sum ≥ s."""
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right := left + 1, s - self.tree[left])

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(leaf, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_stored = min(self.n_stored + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def get(self, s: float):
        """Return (leaf_idx, priority, data) for the sample drawn at value s."""
        leaf_idx  = self._retrieve(0, s)
        data_idx  = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]


# ── PrioritizedReplayBuffer ───────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Experience replay buffer with priority-based sampling.

    Parameters
    ----------
    capacity      : max number of transitions stored
    alpha         : how much prioritization to use  (0 = uniform, 1 = full)
    beta_start    : initial IS correction exponent  (0 = no correction)
    beta_end      : final IS correction exponent    (1 = full correction)
    beta_steps    : number of add() calls over which β is annealed
    epsilon       : small constant to prevent zero priority
    """

    def __init__(
        self,
        capacity:   int   = 100_000,
        alpha:      float = 0.6,
        beta_start: float = 0.4,
        beta_end:   float = 1.0,
        beta_steps: int   = 200_000,
        epsilon:    float = 1e-6,
    ):
        self.tree       = SumTree(capacity)
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta_start
        self._beta_end  = beta_end
        self._beta_inc  = (beta_end - beta_start) / max(beta_steps, 1)
        self.epsilon    = epsilon
        self._max_prio  = 1.0      # track max priority for new experiences

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.tree.n_stored

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a transition with maximum priority (will be corrected later)."""
        self.tree.add(self._max_prio, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Returns
        -------
        states, actions, rewards, next_states, dones  — numpy arrays
        weights                                        — IS weights (float32)
        indices                                        — leaf indices for priority update
        """
        assert len(self) >= batch_size, "Buffer smaller than batch_size"

        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(max(s, 1e-9))
            indices.append(idx)
            priorities.append(prio)
            batch.append(data)

        # IS weights
        n     = len(self)
        probs = np.array(priorities, dtype=np.float64) / (self.tree.total + 1e-9)
        w     = (n * probs) ** (-self.beta)
        w     = (w / w.max()).astype(np.float32)

        # Anneal β
        self.beta = min(self._beta_end, self.beta + self._beta_inc)

        states, actions, rewards, next_states, dones = map(
            lambda t: np.array(t, dtype=np.float32), zip(*batch)
        )
        actions = actions.astype(np.int64)
        dones   = dones.astype(np.float32)

        return states, actions, rewards, next_states, dones, w, indices

    def update_priorities(
        self, indices: list[int], td_errors: np.ndarray
    ) -> None:
        """Recompute priorities from fresh TD errors after a learning step."""
        for idx, err in zip(indices, td_errors):
            prio = (float(abs(err)) + self.epsilon) ** self.alpha
            self._max_prio = max(self._max_prio, prio)
            self.tree.update(idx, prio)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {"beta": self.beta, "max_prio": self._max_prio}

    def load_state_dict(self, d: dict) -> None:
        self.beta       = d.get("beta",     self.beta)
        self._max_prio  = d.get("max_prio", self._max_prio)
