"""
V2 Advanced XAUUSD Trading Environment
=======================================
Key upgrades over environment_advanced.py:

Observation (1 368 features)
─────────────────────────────
  Three CNN-LSTM branches each receive a raw candle window:
    1H : last 100 bars × 8 features
    4H : last  50 bars × 8 features
    1D : last  20 bars × 8 features
  Per-bar features (8):
    open, high, low, close  — price-normalised (÷ most-recent close in window)
    volume                  — max-normalised within window
    atr_14                  — normalised (÷ current close)
    price_pos_50            — close rank in 50-bar high/low range  [0, 1]
    vol_regime              — current ATR ÷ 20-bar mean ATR  [0, 1]
  Portfolio state (8 features) — identical to V1.

Action space (17 total)
────────────────────────
  0          : HOLD
  1 … 16     : (order_type, entry_offset_atr, tp_atr_mult)
    order_type       : limit_buy | limit_sell | stop_buy | stop_sell
    entry_offset_atr : 0.5 × ATR  or  1.0 × ATR  from current close
    tp_atr_mult      : 2.0  or  3.0  (TP = mult × ATR from entry)
    sl always        : 1.5 × ATR  (adapts to current volatility)

Reward shaping improvements
────────────────────────────
  TP hit                 : +tp_mult   (proportional to R:R achieved)
  SL hit                 : −1.0
  SL within 3 candles    : extra −0.5  (penalise bad entry timing)
  Overtrading (max slots): −0.02  (discourage firing when book is full)
  Quiet-market flat book : +0.005 / step  (reward sitting out chop)
  Pending expired        : −0.05
  Open slots forced-close: proportional PnL reward
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ── V2 feature columns ────────────────────────────────────────────────────────
V2_COLS = ["open", "high", "low", "close", "volume",
           "atr_14", "price_pos_50", "vol_regime"]
N_FEAT = len(V2_COLS)          # 8
PORTFOLIO_DIM = 8

# ── Action table ──────────────────────────────────────────────────────────────
_ORDER_TYPES  = ["limit_buy", "limit_sell", "stop_buy", "stop_sell"]
_ENTRY_OFFSETS = config.V2_ENTRY_OFFSETS_ATR   # [0.5, 1.0]
_TP_MULTS      = config.V2_TP_ATR_MULTS        # [2.0, 3.0]

V2_ACTION_TABLE: List[Optional[Tuple]] = [None]    # index 0 = HOLD
for _ot in _ORDER_TYPES:
    for _eo in _ENTRY_OFFSETS:
        for _tp in _TP_MULTS:
            V2_ACTION_TABLE.append((_ot, _eo, _tp))

V2_ACTION_DIM = len(V2_ACTION_TABLE)   # 17


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PendingOrder:
    order_type:  str
    entry_price: float
    sl_price:    float
    tp_price:    float
    lot_size:    float
    risk_amount: float
    placed_at:   int
    tp_mult:     float


@dataclass
class OpenPosition:
    direction:   str      # 'long' | 'short'
    entry_price: float
    sl_price:    float
    tp_price:    float
    lot_size:    float
    risk_amount: float
    tp_mult:     float
    opened_at:   int      # 1H step index when filled

    def unrealised_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) * self.lot_size
        return (self.entry_price - price) * self.lot_size


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnvV2:
    """
    V2 XAUUSD environment: ATR-based SL/TP, raw-feature observations,
    and improved reward shaping.
    """

    def __init__(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        initial_balance: float = config.INITIAL_BALANCE,
        risk_pct:        float = config.ADV_RISK_PCT,
        max_lot:         float = config.ADV_MAX_LOT,
        max_pending:     int   = config.ADV_MAX_PENDING,
        max_open:        int   = config.ADV_MAX_OPEN,
        order_expiry:    int   = config.ADV_ORDER_EXPIRY,
        sl_atr:          float = config.V2_SL_ATR_MULT,
    ):
        self.df_1h  = df_1h
        self.df_4h  = df_4h
        self.df_1d  = df_1d

        self.initial_balance = initial_balance
        self.risk_pct    = risk_pct
        self.max_lot     = max_lot
        self.max_pending = max_pending
        self.max_open    = max_open
        self.order_expiry = order_expiry
        self.sl_atr      = sl_atr

        # Window sizes
        self.w1h = config.V2_WINDOW_1H   # 100
        self.w4h = config.V2_WINDOW_4H   # 50
        self.w1d = config.V2_WINDOW_1D   # 20

        self.obs_dim    = (self.w1h + self.w4h + self.w1d) * N_FEAT + PORTFOLIO_DIM
        self.action_dim = V2_ACTION_DIM

        # Validate required columns
        for col in V2_COLS:
            assert col in df_1h.columns, f"df_1h missing column '{col}'"
            assert col in df_4h.columns, f"df_4h missing column '{col}'"
            assert col in df_1d.columns, f"df_1d missing column '{col}'"

        self._n_1h = len(df_1h)

        # Precompute timestamp alignment (no lookahead)
        ts_1h = df_1h.index.view("int64")
        ts_4h = df_4h.index.view("int64")
        ts_1d = df_1d.index.view("int64")
        self._cnt_4h = np.searchsorted(ts_4h, ts_1h, side="left")
        self._cnt_1d = np.searchsorted(ts_1d, ts_1h, side="left")

        # Minimum step: enough bars in all three TF windows
        ok_1h = self.w1h
        ok_4h = int(np.argmax(self._cnt_4h >= self.w4h)) if (self._cnt_4h >= self.w4h).any() else self.w4h * 4
        ok_1d = int(np.argmax(self._cnt_1d >= self.w1d)) if (self._cnt_1d >= self.w1d).any() else self.w1d * 24
        self.min_step = max(ok_1h, ok_4h, ok_1d) + 1

        self._reset_state(self.min_step, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        start_step: Optional[int] = None,
        max_steps:  Optional[int] = None,
    ) -> np.ndarray:
        s = start_step if start_step is not None else self.min_step
        self._reset_state(s, max_steps)
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < V2_ACTION_DIM, f"Invalid action {action}"

        reward = 0.0
        row    = self.df_1h.iloc[self.current_step]
        high   = float(row["high"])
        low    = float(row["low"])
        close  = float(row["close"])

        # ── 1. Fill pending orders ────────────────────────────────────────────
        still_pending = []
        for order in self.pending_orders:
            filled = (
                (order.order_type == "limit_buy"  and low  <= order.entry_price) or
                (order.order_type == "limit_sell" and high >= order.entry_price) or
                (order.order_type == "stop_buy"   and high >= order.entry_price) or
                (order.order_type == "stop_sell"  and low  <= order.entry_price)
            )
            if filled and len(self.open_positions) < self.max_open:
                direction = "long" if order.order_type in ("limit_buy", "stop_buy") else "short"
                self.open_positions.append(OpenPosition(
                    direction   = direction,
                    entry_price = order.entry_price,
                    sl_price    = order.sl_price,
                    tp_price    = order.tp_price,
                    lot_size    = order.lot_size,
                    risk_amount = order.risk_amount,
                    tp_mult     = order.tp_mult,
                    opened_at   = self.current_step,
                ))
            elif filled:
                reward -= 0.01   # slot full at fill time
            else:
                still_pending.append(order)
        self.pending_orders = still_pending

        # ── 2. SL/TP for open positions ───────────────────────────────────────
        remaining = []
        for pos in self.open_positions:
            closed = False
            if pos.direction == "long":
                if high >= pos.tp_price:
                    pnl    = pos.lot_size * (pos.tp_price - pos.entry_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    self._log_trade(pos, "TP", pos.tp_price, pnl)
                    self.balance += pnl
                    closed = True
                elif low <= pos.sl_price:
                    pnl    = pos.lot_size * (pos.sl_price - pos.entry_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    if self.current_step - pos.opened_at <= config.V2_QUICK_SL_BARS:
                        reward -= config.V2_QUICK_SL_PENALTY   # bad entry timing
                    self._log_trade(pos, "SL", pos.sl_price, pnl)
                    self.balance += pnl
                    closed = True
            else:   # short
                if low <= pos.tp_price:
                    pnl    = pos.lot_size * (pos.entry_price - pos.tp_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    self._log_trade(pos, "TP", pos.tp_price, pnl)
                    self.balance += pnl
                    closed = True
                elif high >= pos.sl_price:
                    pnl    = pos.lot_size * (pos.entry_price - pos.sl_price)
                    reward += pnl / (pos.risk_amount + 1e-9)
                    if self.current_step - pos.opened_at <= config.V2_QUICK_SL_BARS:
                        reward -= config.V2_QUICK_SL_PENALTY
                    self._log_trade(pos, "SL", pos.sl_price, pnl)
                    self.balance += pnl
                    closed = True
            if not closed:
                remaining.append(pos)
        self.open_positions = remaining

        # ── 3. Expire pending orders ──────────────────────────────────────────
        fresh = []
        for order in self.pending_orders:
            if self.current_step - order.placed_at >= self.order_expiry:
                reward -= 0.05
            else:
                fresh.append(order)
        self.pending_orders = fresh

        # ── 4. Place new order or penalise overtrading ────────────────────────
        if action != 0 and V2_ACTION_TABLE[action] is not None:
            can_place = (len(self.pending_orders) < self.max_pending and
                         len(self.open_positions) < self.max_open)
            if can_place:
                ot, eo_atr, tp_mult = V2_ACTION_TABLE[action]
                atr_now = float(self.df_1h.iloc[self.current_step]["atr_14"])
                self._place_order(ot, eo_atr, tp_mult, close, atr_now)
            else:
                reward -= config.V2_OVERTRADE_PENALTY   # overtrading penalty

        # ── 5. Quiet-market flat-book reward ──────────────────────────────────
        if len(self.open_positions) == 0 and len(self.pending_orders) == 0:
            vol_r = float(self.df_1h.iloc[self.current_step]["vol_regime"])
            if vol_r < config.V2_CHOPPY_REGIME_THR:
                reward += config.V2_CHOPPY_HOLD_REWARD

        # ── 6. Advance step ───────────────────────────────────────────────────
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
            "step":      self.current_step,
            "balance":   self.balance,
            "equity":    self._equity(close),
            "n_open":    len(self.open_positions),
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

    def _reset_state(self, start_step: int, max_steps: Optional[int]) -> None:
        self.current_step    = start_step
        self._max_steps      = max_steps
        self._steps_this_ep  = 0
        self.balance         = self.initial_balance
        self.pending_orders: List[PendingOrder]  = []
        self.open_positions: List[OpenPosition]  = []
        self.trade_log       = []

    def _equity(self, price: float) -> float:
        return self.balance + sum(p.unrealised_pnl(price)
                                  for p in self.open_positions)

    def _place_order(
        self,
        order_type: str,
        entry_off_atr: float,
        tp_mult: float,
        close: float,
        atr: float,
    ) -> None:
        is_long = order_type in ("limit_buy", "stop_buy")
        offset  = entry_off_atr * atr

        if order_type == "limit_buy":
            entry = close - offset
        elif order_type == "limit_sell":
            entry = close + offset
        elif order_type == "stop_buy":
            entry = close + offset
        else:   # stop_sell
            entry = close - offset

        sl_dist = self.sl_atr * atr
        tp_dist = tp_mult * atr

        if is_long:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        risk_amount = self.risk_pct * self.balance
        lot_size    = min(risk_amount / (sl_dist + 1e-9), self.max_lot)

        self.pending_orders.append(PendingOrder(
            order_type  = order_type,
            entry_price = entry,
            sl_price    = sl_price,
            tp_price    = tp_price,
            lot_size    = lot_size,
            risk_amount = risk_amount,
            placed_at   = self.current_step,
            tp_mult     = tp_mult,
        ))
        logger.debug("V2 placed %s: entry=%.2f SL=%.2f TP=%.2f lot=%.2f atr=%.2f",
                     order_type, entry, sl_price, tp_price, lot_size, atr)

    def _get_obs(self) -> np.ndarray:
        i = self.current_step

        # ── 1H window ─────────────────────────────────────────────────────────
        obs_1h = self._normalise_window(
            self.df_1h.iloc[i - self.w1h: i]
        )

        # ── 4H window ─────────────────────────────────────────────────────────
        end_4h = int(self._cnt_4h[i])
        if end_4h >= self.w4h:
            obs_4h = self._normalise_window(
                self.df_4h.iloc[end_4h - self.w4h: end_4h]
            )
        else:
            obs_4h = np.zeros(self.w4h * N_FEAT, dtype=np.float32)

        # ── 1D window ─────────────────────────────────────────────────────────
        end_1d = int(self._cnt_1d[i])
        if end_1d >= self.w1d:
            obs_1d = self._normalise_window(
                self.df_1d.iloc[end_1d - self.w1d: end_1d]
            )
        else:
            obs_1d = np.zeros(self.w1d * N_FEAT, dtype=np.float32)

        # ── Portfolio state (8 features) ──────────────────────────────────────
        close  = float(self.df_1h.iloc[i]["close"])
        n_pend = len(self.pending_orders)
        n_open = len(self.open_positions)
        upnl   = sum(p.unrealised_pnl(close) for p in self.open_positions)
        n_long = sum(1 for p in self.open_positions if p.direction == "long")
        n_short = n_open - n_long
        oldest = (max((i - o.placed_at for o in self.pending_orders), default=0)
                  / self.order_expiry)

        portfolio = np.array([
            self.balance / self.initial_balance,
            n_pend / self.max_pending,
            n_open / self.max_open,
            np.clip(upnl / (self.balance + 1e-9), -1.0, 1.0),
            n_pend * self.risk_pct,
            (n_long - n_short) / (self.max_open + 1e-9),
            oldest,
            float(n_pend < self.max_pending and n_open < self.max_open),
        ], dtype=np.float32)

        return np.concatenate([obs_1h, obs_4h, obs_1d, portfolio])

    @staticmethod
    def _normalise_window(window: pd.DataFrame) -> np.ndarray:
        """
        Return a flat float32 array for one timeframe window.

        Normalisation:
          open, high, low, close  : ÷ most-recent close
          volume                  : ÷ max volume in window
          atr_14                  : ÷ most-recent close
          price_pos_50            : already [0, 1]
          vol_regime              : already [0, 1]
        """
        arr = window[V2_COLS].values.copy().astype(np.float32)
        ref = arr[-1, 3] + 1e-9    # most-recent close (index 3 = "close")

        # OHLC columns (0-3)
        arr[:, 0] /= ref
        arr[:, 1] /= ref
        arr[:, 2] /= ref
        arr[:, 3] /= ref
        # Volume (4)
        vol_max = arr[:, 4].max() + 1e-9
        arr[:, 4] /= vol_max
        # ATR (5)
        arr[:, 5] /= ref
        # price_pos_50 (6) — no change
        # vol_regime (7) — no change

        return arr.flatten()

    def _log_trade(self, pos: OpenPosition, exit_type: str,
                   exit_price: float, pnl: float) -> None:
        bars_held = self.current_step - pos.opened_at
        self.trade_log.append({
            "step":       self.current_step,
            "direction":  pos.direction,
            "entry":      pos.entry_price,
            "exit_price": exit_price,
            "exit":       exit_type,
            "pnl":        pnl,
            "lot_size":   pos.lot_size,
            "opened_at":  pos.opened_at,
            "bars_held":  bars_held,
        })
        logger.debug("[%s] %s pnl=%.2f bars=%d", exit_type, pos.direction,
                     pnl, bars_held)
