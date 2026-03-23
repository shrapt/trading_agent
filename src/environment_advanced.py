"""
Advanced XAUUSD Trading Environment
=====================================
Implements professional trading mechanics:
  • Limit and stop orders only (no market orders)
  • Every order requires SL and TP set at placement — immutable
  • Multiple simultaneous pending + open orders
  • 1% risk-per-trade position sizing
  • Multi-timeframe state: 1H (entry timing) + 4H (direction) + 1D (trend)

Action Space (discrete, 33 total)
----------------------------------
  0        : HOLD — place no new order this step
  1 … 32   : place one order = (order_type, entry_offset, sl_pct, tp_ratio)
             order_type  : limit_buy | limit_sell | stop_buy | stop_sell
             entry_offset: 0.2% or 0.5% from current close
             sl_pct      : 0.3% or 0.6% from entry
             tp_ratio    : 1.5× or 2.5× the SL distance

Order fill rules (using candle High/Low, no lookahead)
------------------------------------------------------
  limit_buy   : fills when LOW  ≤ entry  (buying the dip)
  limit_sell  : fills when HIGH ≥ entry  (selling the rally)
  stop_buy    : fills when HIGH ≥ entry  (momentum breakout long)
  stop_sell   : fills when LOW  ≤ entry  (momentum breakout short)

SL/TP execution (checked every step, immutable once set)
---------------------------------------------------------
  Long  position: TP when HIGH ≥ tp_price; SL when LOW  ≤ sl_price
  Short position: TP when LOW  ≤ tp_price; SL when HIGH ≥ sl_price
  (TP checked first — agent-favorable)

Reward signal (normalised to units of R = 1% of balance)
---------------------------------------------------------
  TP hit          : +tp_ratio  (e.g. +1.5 or +2.5)
  SL hit          : -1.0
  Pending expired : -0.05  (small penalty for tying up order slots)
  Per step        :  0.0   (no noise; rewards come from trade outcomes)

State (393 features)
--------------------
  [0:385]   MTF market features  (1H×20 + 4H×10 + 1D×5) × 11 features
  [385:393] Portfolio state (8 features):
              balance / initial_balance
              n_pending / MAX_PENDING
              n_open    / MAX_OPEN
              total_unrealised_pnl / balance
              total_risk_pct  (sum of each open position's 1% risk exposure)
              net_direction   (-1=full short … +1=full long, normalised)
              oldest_pending_age / ORDER_EXPIRY
              can_place_new   (1.0 if slots available, else 0.0)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data import normalise_window, FEATURE_COLS
from src import config

logger = logging.getLogger(__name__)

N_FEAT = len(FEATURE_COLS)   # 11

# ── Action table ──────────────────────────────────────────────────────────────
# Each entry: (order_type, entry_offset_pct, sl_pct, tp_ratio)
# Index 0 is HOLD (None).

ORDER_TYPES    = ["limit_buy", "limit_sell", "stop_buy", "stop_sell"]
ENTRY_OFFSETS  = config.ADV_ENTRY_OFFSETS   # [0.002, 0.005]
SL_PCTS        = config.ADV_SL_PCTS         # [0.003, 0.006]
TP_RATIOS      = config.ADV_TP_RATIOS       # [1.5, 2.5]

ACTION_TABLE: List[Optional[Tuple]] = [None]   # index 0 = HOLD
for _ot in ORDER_TYPES:
    for _eo in ENTRY_OFFSETS:
        for _sl in SL_PCTS:
            for _tp in TP_RATIOS:
                ACTION_TABLE.append((_ot, _eo, _sl, _tp))

ACTION_DIM = len(ACTION_TABLE)   # 33
PORTFOLIO_FEAT = 8


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PendingOrder:
    order_type:  str     # 'limit_buy' | 'limit_sell' | 'stop_buy' | 'stop_sell'
    entry_price: float
    sl_price:    float
    tp_price:    float
    lot_size:    float
    risk_amount: float   # amount at risk (for reward normalisation)
    placed_at:   int     # step index when placed
    tp_ratio:    float   # stored for logging


@dataclass
class OpenPosition:
    direction:   str     # 'long' | 'short'
    entry_price: float
    sl_price:    float
    tp_price:    float
    lot_size:    float
    risk_amount: float
    tp_ratio:    float
    opened_at:   int

    def unrealised_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.lot_size
        return (self.entry_price - current_price) * self.lot_size


# ── Environment ───────────────────────────────────────────────────────────────

class AdvancedTradingEnv:
    """
    Advanced XAUUSD environment with limit/stop orders, immutable SL/TP,
    1% risk sizing, multiple simultaneous orders, and MTF state.
    """

    def __init__(self,
                 df_1h: pd.DataFrame,
                 df_4h: pd.DataFrame,
                 df_1d: pd.DataFrame,
                 window_1h: int   = config.MTF_WINDOW_1H,   # 20
                 window_4h: int   = config.MTF_WINDOW_4H,   # 10
                 window_1d: int   = config.MTF_WINDOW_1D,   # 5
                 initial_balance: float = config.INITIAL_BALANCE,
                 risk_pct: float  = config.ADV_RISK_PCT,    # 0.01
                 max_lot: float   = config.ADV_MAX_LOT,
                 max_pending: int = config.ADV_MAX_PENDING,
                 max_open: int    = config.ADV_MAX_OPEN,
                 order_expiry: int = config.ADV_ORDER_EXPIRY):

        self.df_1h  = df_1h
        self.df_4h  = df_4h
        self.df_1d  = df_1d
        self.w1h    = window_1h
        self.w4h    = window_4h
        self.w1d    = window_1d

        self.initial_balance = initial_balance
        self.risk_pct    = risk_pct
        self.max_lot     = max_lot
        self.max_pending = max_pending
        self.max_open    = max_open
        self.order_expiry = order_expiry

        self.obs_dim = (window_1h + window_4h + window_1d) * N_FEAT + PORTFOLIO_FEAT
        self.action_dim = ACTION_DIM   # 33

        # Minimum step: enough history in all three timeframes
        self.min_step = max(window_1h, window_4h * 4, window_1d * 24) + 1

        # Precompute timestamp alignment (no lookahead)
        ts_1h = df_1h.index.view("int64")
        ts_4h = df_4h.index.view("int64")
        ts_1d = df_1d.index.view("int64")
        self._cnt_4h = np.searchsorted(ts_4h, ts_1h, side="left")
        self._cnt_1d = np.searchsorted(ts_1d, ts_1h, side="left")

        self._n_1h  = len(df_1h)
        self._reset_state(self.min_step, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, start_step: Optional[int] = None,
              max_steps: Optional[int] = None) -> np.ndarray:
        s = start_step if start_step is not None else self.min_step
        self._reset_state(s, max_steps)
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < ACTION_DIM, f"Invalid action {action}"

        reward = 0.0
        row    = self.df_1h.iloc[self.current_step]
        high   = float(row["high"])
        low    = float(row["low"])
        close  = float(row["close"])

        # 1. Check fills for pending orders (using this candle's H/L)
        still_pending = []
        for order in self.pending_orders:
            filled = False
            if order.order_type == "limit_buy"  and low  <= order.entry_price:
                filled = True
            elif order.order_type == "limit_sell" and high >= order.entry_price:
                filled = True
            elif order.order_type == "stop_buy"   and high >= order.entry_price:
                filled = True
            elif order.order_type == "stop_sell"  and low  <= order.entry_price:
                filled = True

            if filled and len(self.open_positions) < self.max_open:
                direction = ("long"
                             if order.order_type in ("limit_buy", "stop_buy")
                             else "short")
                pos = OpenPosition(
                    direction   = direction,
                    entry_price = order.entry_price,
                    sl_price    = order.sl_price,
                    tp_price    = order.tp_price,
                    lot_size    = order.lot_size,
                    risk_amount = order.risk_amount,
                    tp_ratio    = order.tp_ratio,
                    opened_at   = self.current_step,
                )
                self.open_positions.append(pos)
                logger.debug("Order filled: %s @ %.2f", order.order_type,
                             order.entry_price)
            elif filled:
                # Slots full — cancel rather than leave pending
                reward -= 0.01
            else:
                still_pending.append(order)
        self.pending_orders = still_pending

        # 2. Check SL/TP for open positions (TP checked first)
        remaining_positions = []
        for pos in self.open_positions:
            closed = False
            if pos.direction == "long":
                if high >= pos.tp_price:
                    pnl     = pos.lot_size * (pos.tp_price - pos.entry_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    self._log_trade(pos, "TP", pos.tp_price, pnl)
                    self.balance += pnl
                    closed = True
                elif low <= pos.sl_price:
                    pnl     = pos.lot_size * (pos.sl_price - pos.entry_price)
                    reward += pnl / (pos.risk_amount + 1e-9)  # ≈ -1
                    self._log_trade(pos, "SL", pos.sl_price, pnl)
                    self.balance += pnl
                    closed = True
            else:  # short
                if low <= pos.tp_price:
                    pnl     = pos.lot_size * (pos.entry_price - pos.tp_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    self._log_trade(pos, "TP", pos.tp_price, pnl)
                    self.balance += pnl
                    closed = True
                elif high >= pos.sl_price:
                    pnl     = pos.lot_size * (pos.entry_price - pos.sl_price)
                    reward += pnl / (pos.risk_amount + 1e-9)  # ≈ -1
                    self._log_trade(pos, "SL", pos.sl_price, pnl)
                    self.balance += pnl
                    closed = True
            if not closed:
                remaining_positions.append(pos)
        self.open_positions = remaining_positions

        # 3. Expire old pending orders
        fresh = []
        for order in self.pending_orders:
            age = self.current_step - order.placed_at
            if age >= self.order_expiry:
                reward -= 0.05   # penalty for unfilled order
                logger.debug("Pending expired: %s @ %.2f (age %d)",
                             order.order_type, order.entry_price, age)
            else:
                fresh.append(order)
        self.pending_orders = fresh

        # 4. Execute agent action (place new order if slots available)
        if action != 0 and ACTION_TABLE[action] is not None:
            can_place = (len(self.pending_orders) < self.max_pending and
                         len(self.open_positions) < self.max_open)
            if can_place:
                order_type, entry_off, sl_pct, tp_ratio = ACTION_TABLE[action]
                self._place_order(order_type, entry_off, sl_pct, tp_ratio, close)

        # 5. Advance step
        self.current_step   += 1
        self._steps_this_ep += 1
        at_end = self.current_step >= self._n_1h - 1
        at_lim = (self._max_steps is not None and
                  self._steps_this_ep >= self._max_steps)
        done   = at_end or at_lim

        # Force-close all positions at episode end
        if done:
            end_price = float(self.df_1h.iloc[self.current_step]["close"])
            for pos in self.open_positions:
                if pos.direction == "long":
                    pnl = pos.lot_size * (end_price - pos.entry_price)
                else:
                    pnl = pos.lot_size * (pos.entry_price - end_price)
                reward += pnl / (pos.risk_amount + 1e-9)
                self._log_trade(pos, "FORCED", end_price, pnl)
                self.balance += pnl
            self.open_positions = []
            self.pending_orders = []

        obs  = self._get_obs()
        info = {
            "step":     self.current_step,
            "balance":  self.balance,
            "equity":   self._equity(close),
            "n_open":   len(self.open_positions),
            "n_pending": len(self.pending_orders),
        }
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
            "win_rate":      len(wins) / max(len(pnls), 1),
            "total_pnl":     sum(pnls),
            "avg_win":       float(np.mean(wins))   if wins   else 0.0,
            "avg_loss":      float(np.mean(losses)) if losses else 0.0,
            "profit_factor": abs(sum(wins) / (sum(losses) - 1e-9))
                             if losses else float("inf"),
            "final_balance": self.balance,
            "tp_count":      sum(1 for t in self.trade_log if t["exit"] == "TP"),
            "sl_count":      sum(1 for t in self.trade_log if t["exit"] == "SL"),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_state(self, start_step: int,
                     max_steps: Optional[int]) -> None:
        self.current_step    = start_step
        self._max_steps      = max_steps
        self._steps_this_ep  = 0
        self.balance         = self.initial_balance
        self.pending_orders: List[PendingOrder]  = []
        self.open_positions: List[OpenPosition]  = []
        self.trade_log       = []

    def _equity(self, current_price: float) -> float:
        upnl = sum(p.unrealised_pnl(current_price) for p in self.open_positions)
        return self.balance + upnl

    def _place_order(self, order_type: str, entry_off: float,
                     sl_pct: float, tp_ratio: float, close: float) -> None:
        """Compute entry / SL / TP prices and lot size, then queue the order."""
        is_long = order_type in ("limit_buy", "stop_buy")

        if order_type == "limit_buy":
            entry = close * (1.0 - entry_off)
        elif order_type == "limit_sell":
            entry = close * (1.0 + entry_off)
        elif order_type == "stop_buy":
            entry = close * (1.0 + entry_off)
        else:  # stop_sell
            entry = close * (1.0 - entry_off)

        sl_dist = entry * sl_pct
        if is_long:
            sl_price = entry - sl_dist
            tp_price = entry + sl_dist * tp_ratio
        else:
            sl_price = entry + sl_dist
            tp_price = entry - sl_dist * tp_ratio

        risk_amount = self.risk_pct * self.balance
        lot_size    = min(risk_amount / (sl_dist + 1e-9), self.max_lot)

        order = PendingOrder(
            order_type  = order_type,
            entry_price = entry,
            sl_price    = sl_price,
            tp_price    = tp_price,
            lot_size    = lot_size,
            risk_amount = risk_amount,
            placed_at   = self.current_step,
            tp_ratio    = tp_ratio,
        )
        self.pending_orders.append(order)
        logger.debug("Placed %s: entry=%.2f SL=%.2f TP=%.2f lot=%.2f",
                     order_type, entry, sl_price, tp_price, lot_size)

    def _get_obs(self) -> np.ndarray:
        i = self.current_step

        # ── MTF market state ─────────────────────────────────────────────────
        w1h    = self.df_1h.iloc[i - self.w1h : i]
        obs_1h = normalise_window(w1h)

        end_4h = self._cnt_4h[i]
        if end_4h >= self.w4h:
            obs_4h = normalise_window(self.df_4h.iloc[end_4h - self.w4h : end_4h])
        else:
            obs_4h = np.zeros(self.w4h * N_FEAT, dtype=np.float32)

        end_1d = self._cnt_1d[i]
        if end_1d >= self.w1d:
            obs_1d = normalise_window(self.df_1d.iloc[end_1d - self.w1d : end_1d])
        else:
            obs_1d = np.zeros(self.w1d * N_FEAT, dtype=np.float32)

        # ── Portfolio state (8 features) ─────────────────────────────────────
        close    = float(self.df_1h.iloc[i]["close"])
        n_pend   = len(self.pending_orders)
        n_open   = len(self.open_positions)
        upnl     = sum(p.unrealised_pnl(close) for p in self.open_positions)
        n_long   = sum(1 for p in self.open_positions if p.direction == "long")
        n_short  = n_open - n_long
        oldest   = (max((i - o.placed_at for o in self.pending_orders), default=0)
                    / self.order_expiry)

        portfolio = np.array([
            self.balance / self.initial_balance,
            n_pend / self.max_pending,
            n_open / self.max_open,
            np.clip(upnl / (self.balance + 1e-9), -1.0, 1.0),
            n_pend * self.risk_pct,                    # total pending exposure
            (n_long - n_short) / (self.max_open + 1e-9),
            oldest,
            float(n_pend < self.max_pending and n_open < self.max_open),
        ], dtype=np.float32)

        return np.concatenate([obs_1h, obs_4h, obs_1d, portfolio])

    def _log_trade(self, pos: OpenPosition, exit_type: str,
                   exit_price: float, pnl: float) -> None:
        self.trade_log.append({
            "step":       self.current_step,
            "direction":  pos.direction,
            "entry":      pos.entry_price,
            "exit_price": exit_price,
            "exit":       exit_type,
            "pnl":        pnl,
            "lot_size":   pos.lot_size,
        })
        logger.debug("[%s] %s pnl=%.2f @ %.2f", exit_type, pos.direction,
                     pnl, exit_price)
