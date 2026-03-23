"""
XAUUSD Trading Environment
===========================
A gym-style environment for paper-trading Gold (XAUUSD).

Actions
-------
  0 – HOLD   : do nothing
  1 – BUY    : open a long position (or ignore if already long)
  2 – SELL   : open a short position (or ignore if already short)

If the agent holds a position opposite to the new action, the current
position is closed first, then the new one is opened.

Reward
------
  Shaped reward = realised PnL / entry_price   (on close)
               + small step reward from unrealised PnL change
               - inaction penalty when holding a losing trade too long
  This pushes the agent to: cut losses fast, let winners run.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data import normalise_window, FEATURE_COLS
from src import config

logger = logging.getLogger(__name__)

# Action constants
HOLD = 0
BUY  = 1
SELL = 2


class Position:
    """Tracks a single open trade."""
    def __init__(self, direction: int, entry_price: float,
                 lot_size: float, spread: float):
        self.direction   = direction      # BUY or SELL
        self.entry_price = entry_price + (spread if direction == BUY else -spread)
        self.lot_size    = lot_size
        self.pnl         = 0.0

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == BUY:
            return (current_price - self.entry_price) * self.lot_size
        else:
            return (self.entry_price - current_price) * self.lot_size

    def close(self, current_price: float, spread: float) -> float:
        exit_price = current_price - (spread if self.direction == BUY else -spread)
        if self.direction == BUY:
            self.pnl = (exit_price - self.entry_price) * self.lot_size
        else:
            self.pnl = (self.entry_price - exit_price) * self.lot_size
        return self.pnl


class XAUUSDEnv:
    """
    Paper-trading environment for XAUUSD.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicator data (output of data.fetch_data).
    window_size : int
        Number of past candles in the observation.
    initial_balance : float
        Starting account equity.
    lot_size : float
        Ounces of gold per trade.
    spread : float
        Simulated spread per ounce.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int  = config.WINDOW_SIZE,
                 initial_balance: float = config.INITIAL_BALANCE,
                 lot_size: float   = config.LOT_SIZE,
                 spread: float     = config.SPREAD_PIPS):
        self.df              = df.reset_index(drop=True)
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.lot_size        = lot_size
        self.spread          = spread

        self.n_features = len(FEATURE_COLS)
        self.obs_dim    = window_size * self.n_features

        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one time-step."""
        assert action in (HOLD, BUY, SELL), f"Invalid action {action}"

        prev_equity = self._equity()
        reward = 0.0

        current_price = self.df.loc[self.current_step, "close"]

        # ── Trade logic ──────────────────────────────────────────────────────
        if self.position is not None:
            same_dir = self.position.direction == action
            if action != HOLD and not same_dir:
                # Close current position
                pnl = self.position.close(current_price, self.spread)
                self.balance += pnl
                rel_pnl = pnl / (self.position.entry_price * self.lot_size)
                reward += rel_pnl * 10           # scale for learning signal
                self._log_trade(action, current_price, pnl)
                self.position = None
                # Immediately open new position in opposite direction
                self.position = Position(action, current_price,
                                         self.lot_size, self.spread)

            elif action == HOLD:
                # Holding: small step reward from unrealised PnL change
                upnl = self.position.unrealised_pnl(current_price)
                upnl_prev = self._prev_unrealised_pnl
                reward += (upnl - upnl_prev) / (self.position.entry_price * self.lot_size + 1e-9)
                self._prev_unrealised_pnl = upnl

                # Inaction penalty: if trade is deep in loss, nudge to cut
                if upnl < -0.005 * self.position.entry_price * self.lot_size:
                    reward -= 0.001

        else:
            if action in (BUY, SELL):
                self.position = Position(action, current_price,
                                         self.lot_size, self.spread)
                self._prev_unrealised_pnl = 0.0

        # ── Advance step ─────────────────────────────────────────────────────
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Force-close at end of episode
        if done and self.position is not None:
            close_price = self.df.loc[self.current_step, "close"]
            pnl = self.position.close(close_price, self.spread)
            self.balance += pnl
            rel_pnl = pnl / (self.position.entry_price * self.lot_size + 1e-9)
            reward += rel_pnl * 10
            self._log_trade(HOLD, close_price, pnl, forced=True)
            self.position = None

        obs    = self._get_obs()
        equity = self._equity()
        info   = {
            "step":    self.current_step,
            "balance": self.balance,
            "equity":  equity,
            "trades":  len(self.trade_log),
        }
        return obs, float(reward), done, info

    def render(self, mode: str = "human") -> None:
        equity = self._equity()
        pos_str = "NONE"
        if self.position:
            cp = self.df.loc[self.current_step, "close"]
            pos_str = (f"{'BUY' if self.position.direction==BUY else 'SELL'} "
                       f"@ {self.position.entry_price:.2f} "
                       f"(uPnL: {self.position.unrealised_pnl(cp):.2f})")
        print(f"Step {self.current_step:5d} | "
              f"Price {self.df.loc[self.current_step,'close']:.2f} | "
              f"Equity {equity:.2f} | Position: {pos_str}")

    def trade_summary(self) -> dict:
        """Return performance statistics over all closed trades."""
        if not self.trade_log:
            return {"total_trades": 0}
        pnls   = [t["pnl"] for t in self.trade_log]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            "total_trades":  len(pnls),
            "win_rate":      len(wins) / len(pnls) if pnls else 0,
            "total_pnl":     sum(pnls),
            "avg_win":       np.mean(wins)   if wins   else 0,
            "avg_loss":      np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / (sum(losses) + 1e-9)),
            "final_balance": self.balance,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.current_step           = self.window_size
        self.balance                = self.initial_balance
        self.position: Optional[Position] = None
        self._prev_unrealised_pnl   = 0.0
        self.trade_log              = []

    def _equity(self) -> float:
        if self.position is None:
            return self.balance
        cp = self.df.loc[self.current_step, "close"]
        return self.balance + self.position.unrealised_pnl(cp)

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end   = self.current_step
        window = self.df.iloc[start:end]
        return normalise_window(window)

    def _log_trade(self, action: int, price: float, pnl: float,
                   forced: bool = False) -> None:
        self.trade_log.append({
            "step":   self.current_step,
            "price":  price,
            "pnl":    pnl,
            "forced": forced,
        })
        result = "WIN" if pnl > 0 else "LOSS"
        logger.debug("Trade closed [%s] pnl=%.2f at price=%.2f%s",
                     result, pnl, price, " (forced)" if forced else "")
