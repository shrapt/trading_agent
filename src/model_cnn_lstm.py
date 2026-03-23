"""
Hybrid CNN-LSTM-Dueling DQN
============================
Architecture for each timeframe branch:
  (batch, seq_len, n_features)
      → permute → Conv1D stack (discovers price patterns automatically)
      → LSTM (learns temporal dependencies)
      → last hidden state

Three branches (1H/4H/1D) are concatenated with a portfolio state vector,
passed through fully-connected fusion layers, then split into Dueling
Value + Advantage streams for Q-value output.

The network deliberately receives *raw* OHLCV plus only 3 minimal
hand-crafted hints so that it can discover patterns (support/resistance,
breakouts, candle structure) without hard-coded feature engineering.
"""

import torch
import torch.nn as nn


# ── Per-timeframe CNN-LSTM branch ─────────────────────────────────────────────

class CNNLSTMBranch(nn.Module):
    """
    1D-CNN → 2-layer LSTM branch.

    CNN kernel progression (3→5→3) gives overlapping receptive fields that
    detect short patterns (3-bar reversal) and medium-range structures (5-bar
    consolidation) simultaneously before the LSTM integrates them over time.

    No BatchNorm: avoids instability when batch_size=1 during action selection.
    """

    def __init__(
        self,
        n_features:   int,
        cnn_channels: int = 32,
        lstm_hidden:  int = 128,
    ):
        super().__init__()

        ch = cnn_channels
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, ch,      kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(ch,         ch * 2,  kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(ch * 2,     ch * 2,  kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size  = ch * 2,
            hidden_size = lstm_hidden,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.1,
        )
        self.out_dim = lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_features)
        x = x.permute(0, 2, 1)          # (batch, n_features, seq_len)
        x = self.conv(x)                 # (batch, ch*2, seq_len)
        x = x.permute(0, 2, 1)          # (batch, seq_len, ch*2)
        _, (h, _) = self.lstm(x)        # h : (num_layers, batch, hidden)
        return h[-1]                     # (batch, hidden)


# ── Full hybrid network ────────────────────────────────────────────────────────

class HybridDQN(nn.Module):
    """
    CNN-LSTM-Dueling DQN with three timeframe branches.

    Observation layout (flat, shape = obs_dim):
      [0       : W_1H*N_FEAT]          1H  window (100 bars × 8 features)
      [W_1H*N  : (W_1H+W_4H)*N_FEAT]  4H  window (50  bars × 8 features)
      [(…)     : (…)+W_1D*N_FEAT]     1D  window (20  bars × 8 features)
      [last 8] : portfolio state

    Features (8 per bar):
      open, high, low, close  — price-normalised (÷ current close)
      volume                  — window-max normalised
      atr_14                  — normalised by current close
      price_pos_50            — where close sits in 50-bar high/low range [0,1]
      vol_regime              — current ATR ÷ 20-bar mean ATR, clipped [0,1]
    """

    W_1H          = 100
    W_4H          = 50
    W_1D          = 20
    N_FEAT        = 8
    PORTFOLIO_DIM = 8

    def __init__(self, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.branch_1h = CNNLSTMBranch(self.N_FEAT, cnn_channels=32, lstm_hidden=128)
        self.branch_4h = CNNLSTMBranch(self.N_FEAT, cnn_channels=32, lstm_hidden=128)
        self.branch_1d = CNNLSTMBranch(self.N_FEAT, cnn_channels=16, lstm_hidden=64)

        # 128 + 128 + 64 + 8 = 328
        fusion_in = (
            self.branch_1h.out_dim
            + self.branch_4h.out_dim
            + self.branch_1d.out_dim
            + self.PORTFOLIO_DIM
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in,  hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )

        # Dueling streams
        self.value     = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    @property
    def obs_dim(self) -> int:
        return (self.W_1H + self.W_4H + self.W_1D) * self.N_FEAT + self.PORTFOLIO_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, obs_dim)
        batch = x.shape[0]
        n1 = self.W_1H * self.N_FEAT     # 800
        n2 = self.W_4H * self.N_FEAT     # 400
        n3 = self.W_1D * self.N_FEAT     # 160

        x_1h = x[:, :n1].view(batch, self.W_1H, self.N_FEAT)
        x_4h = x[:, n1:n1 + n2].view(batch, self.W_4H, self.N_FEAT)
        x_1d = x[:, n1 + n2:n1 + n2 + n3].view(batch, self.W_1D, self.N_FEAT)
        portfolio = x[:, n1 + n2 + n3:]

        f = torch.cat(
            [self.branch_1h(x_1h), self.branch_4h(x_4h),
             self.branch_1d(x_1d), portfolio],
            dim=1,
        )
        h = self.fusion(f)

        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))   # Dueling Q
