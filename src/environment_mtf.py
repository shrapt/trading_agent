"""
Multi-Timeframe Trading Environment
=====================================
Extends the single-timeframe environment with 1D / 4H / 1H context.

State composition
-----------------
  [0  : W1H*11)   last W1H hourly candles     → entry/exit timing
  [W1H*11 : W4H*11) last W4H 4-hour candles  → medium-term direction
  [W4H*11 : W1D*11) last W1D daily candles   → macro trend / big picture

Alignment rule: strictly no lookahead — for a 1H bar at time t, only
4H/1D bars whose open timestamp is STRICTLY BEFORE t are used.

Episode windowing
-----------------
Call reset(start_step=i) to begin an episode at 1H bar i.
If max_steps is set the episode ends after that many 1H steps,
allowing random-window training over the full dataset.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data import normalise_window, FEATURE_COLS
from src import config

logger = logging.getLogger(__name__)

HOLD = 0
BUY  = 1
SELL = 2

N_FEAT = len(FEATURE_COLS)   # 11


class Position:
    """Single open trade (identical to single-TF env)."""
    def __init__(self, direction, entry_price, lot_size, spread):
        self.direction   = direction
        self.entry_price = entry_price + (spread if direction == BUY else -spread)
        self.lot_size    = lot_size

    def unrealised_pnl(self, price):
        if self.direction == BUY:
            return (price - self.entry_price) * self.lot_size
        return (self.entry_price - price) * self.lot_size

    def close(self, price, spread):
        exit_price = price - (spread if self.direction == BUY else -spread)
        if self.direction == BUY:
            pnl = (exit_price - self.entry_price) * self.lot_size
        else:
            pnl = (self.entry_price - exit_price) * self.lot_size
        return pnl


class MultiTimeframeEnv:
    """
    Paper-trading environment with 1H / 4H / 1D multi-timeframe state.

    Parameters
    ----------
    df_1h, df_4h, df_1d : pd.DataFrame
        OHLCV + indicator frames with a DatetimeIndex, sorted ascending.
    window_1h / window_4h / window_1d : int
        Number of bars from each timeframe included in the observation.
    initial_balance, lot_size, spread : float
        Paper-trading parameters.
    """

    def __init__(self,
                 df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame,
                 df_1d: pd.DataFrame,
                 window_1h: int   = config.MTF_WINDOW_1H,
                 window_4h: int   = config.MTF_WINDOW_4H,
                 window_1d: int   = config.MTF_WINDOW_1D,
                 initial_balance: float = config.INITIAL_BALANCE,
                 lot_size: float  = config.LOT_SIZE,
                 spread: float    = config.SPREAD_PIPS):

        # Store frames (keep DatetimeIndex for alignment)
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.df_1d = df_1d

        self.window_1h = window_1h
        self.window_4h = window_4h
        self.window_1d = window_1d

        self.initial_balance = initial_balance
        self.lot_size        = lot_size
        self.spread          = spread

        self.obs_dim = (window_1h + window_4h + window_1d) * N_FEAT  # 385

        # Minimum valid step: enough history in all three timeframes
        # 1H needs window_1h bars behind it
        # 4H: window_4h * 4 hours worth → ~window_4h*4 1H bars
        # 1D: window_1d * 24 hours       → ~window_1d*24 1H bars
        self.min_step = max(window_1h,
                            window_4h * 4,
                            window_1d * 24) + 1

        # Precompute alignment: for each 1H position i, find how many
        # 4H/1D bars have open_ts STRICTLY BEFORE df_1h.index[i].
        # np.searchsorted(..., side='left') returns the first index where
        # ts_tf >= ts_1h[i], so everything to its left is < ts_1h[i].
        ts_1h = df_1h.index.view("int64")   # ns since epoch
        ts_4h = df_4h.index.view("int64")
        ts_1d = df_1d.index.view("int64")
        self._cnt_4h = np.searchsorted(ts_4h, ts_1h, side="left")  # shape (N_1H,)
        self._cnt_1d = np.searchsorted(ts_1d, ts_1h, side="left")  # shape (N_1H,)

        self._n_1h = len(df_1h)
        self._reset_state(self.min_step, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self,
              start_step: Optional[int] = None,
              max_steps: Optional[int]  = None) -> np.ndarray:
        s = start_step if start_step is not None else self.min_step
        self._reset_state(s, max_steps)
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert action in (HOLD, BUY, SELL)

        reward        = 0.0
        current_price = float(self.df_1h.iloc[self.current_step]["close"])

        # ── Trade logic ──────────────────────────────────────────────────────
        if self.position is not None:
            if action != HOLD and self.position.direction != action:
                pnl      = self.position.close(current_price, self.spread)
                self.balance += pnl
                rel_pnl  = pnl / (self.position.entry_price * self.lot_size + 1e-9)
                reward  += rel_pnl * 10
                self._log_trade(current_price, pnl)
                self.position = None
                self.position = Position(action, current_price,
                                         self.lot_size, self.spread)
            elif action == HOLD:
                upnl  = self.position.unrealised_pnl(current_price)
                reward += (upnl - self._prev_upnl) / (
                    self.position.entry_price * self.lot_size + 1e-9)
                self._prev_upnl = upnl
                if upnl < -0.005 * self.position.entry_price * self.lot_size:
                    reward -= 0.001
        else:
            if action in (BUY, SELL):
                self.position    = Position(action, current_price,
                                            self.lot_size, self.spread)
                self._prev_upnl  = 0.0

        # ── Advance ──────────────────────────────────────────────────────────
        self.current_step  += 1
        self._steps_this_ep += 1
        at_data_end  = self.current_step >= self._n_1h - 1
        at_ep_limit  = (self._max_steps is not None and
                        self._steps_this_ep >= self._max_steps)
        done = at_data_end or at_ep_limit

        if done and self.position is not None:
            close_price = float(self.df_1h.iloc[self.current_step]["close"])
            pnl = self.position.close(close_price, self.spread)
            self.balance += pnl
            rel_pnl  = pnl / (self.position.entry_price * self.lot_size + 1e-9)
            reward  += rel_pnl * 10
            self._log_trade(close_price, pnl, forced=True)
            self.position = None

        obs  = self._get_obs()
        info = {"step": self.current_step, "balance": self.balance,
                "equity": self._equity(), "trades": len(self.trade_log)}
        return obs, float(reward), done, info

    def trade_summary(self) -> dict:
        if not self.trade_log:
            return {"total_trades": 0, "total_pnl": 0.0,
                    "win_rate": 0.0, "final_balance": self.balance}
        pnls   = [t["pnl"] for t in self.trade_log]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "total_trades":  len(pnls),
            "win_rate":      len(wins) / len(pnls),
            "total_pnl":     sum(pnls),
            "avg_win":       float(np.mean(wins))   if wins   else 0.0,
            "avg_loss":      float(np.mean(losses)) if losses else 0.0,
            "profit_factor": abs(sum(wins) / (sum(losses) + 1e-9)),
            "final_balance": self.balance,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_state(self, start_step: int, max_steps: Optional[int]) -> None:
        self.current_step    = start_step
        self._max_steps      = max_steps
        self._steps_this_ep  = 0
        self.balance         = self.initial_balance
        self.position        = None
        self._prev_upnl      = 0.0
        self.trade_log       = []

    def _equity(self) -> float:
        if self.position is None:
            return self.balance
        price = float(self.df_1h.iloc[self.current_step]["close"])
        return self.balance + self.position.unrealised_pnl(price)

    def _get_obs(self) -> np.ndarray:
        i = self.current_step

        # ── 1H window ────────────────────────────────────────────────────────
        w1h = self.df_1h.iloc[i - self.window_1h : i]
        obs_1h = normalise_window(w1h)

        # ── 4H window ────────────────────────────────────────────────────────
        # _cnt_4h[i] = number of 4H bars with ts < ts_1h[i]
        end_4h = self._cnt_4h[i]
        if end_4h >= self.window_4h:
            w4h = self.df_4h.iloc[end_4h - self.window_4h : end_4h]
            obs_4h = normalise_window(w4h)
        else:
            # Not enough 4H history yet — pad with zeros
            obs_4h = np.zeros(self.window_4h * N_FEAT, dtype=np.float32)

        # ── 1D window ────────────────────────────────────────────────────────
        end_1d = self._cnt_1d[i]
        if end_1d >= self.window_1d:
            w1d = self.df_1d.iloc[end_1d - self.window_1d : end_1d]
            obs_1d = normalise_window(w1d)
        else:
            obs_1d = np.zeros(self.window_1d * N_FEAT, dtype=np.float32)

        return np.concatenate([obs_1h, obs_4h, obs_1d])

    def _log_trade(self, price: float, pnl: float, forced: bool = False) -> None:
        self.trade_log.append({"step": self.current_step,
                                "price": price, "pnl": pnl, "forced": forced})
        logger.debug("Trade closed pnl=%.2f @ %.2f%s",
                     pnl, price, " (forced)" if forced else "")
