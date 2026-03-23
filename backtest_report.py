"""
Backtest Report — 2023-2025 Out-of-Sample Evaluation
=====================================================
Loads the best model and runs it on 2023-2025 data (never seen during training).

Reports
-------
  total trades, win rate, max drawdown, Sharpe ratio, profit factor,
  avg R:R, monthly PnL breakdown

Output
------
  logs/backtest_equity_curve.png   — equity curve + drawdown chart

Usage
-----
  python backtest_report.py                        # uses ADV_BEST_MODEL_PATH
  python backtest_report.py --model models/foo.pth # custom checkpoint
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

import src.config as cfg

cfg.HIDDEN_DIM = cfg.ADV_HIDDEN_DIM

from src.data import load_historical_csv, add_indicators
from src.environment_advanced import AdvancedTradingEnv, ACTION_DIM
from src.agent import DQNAgent

CHART_PATH = os.path.join(cfg.LOG_DIR, "backtest_equity_curve.png")
VAL_START  = "2023-01-01"


def _load_tf(path: str) -> pd.DataFrame:
    df = load_historical_csv(path)
    df = add_indicators(df)
    df.dropna(inplace=True)
    return df


def run_backtest(model_path: str = None) -> dict:
    if model_path is None:
        model_path = cfg.ADV_BEST_MODEL_PATH

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading 1H data …")
    df_1h = _load_tf(cfg.MTF_DATA_1H)
    logger.info("Loading 4H data …")
    df_4h = _load_tf(cfg.MTF_DATA_4H)
    logger.info("Loading 1D data …")
    df_1d = _load_tf(cfg.MTF_DATA_1D)

    # ── Locate validation window ───────────────────────────────────────────────
    mask = df_1h.index >= VAL_START
    if not mask.any():
        raise ValueError(f"No 1H bars found on or after {VAL_START}")

    val_start_idx = int(np.argmax(mask.values))
    val_end_idx   = len(df_1h) - 2          # leave one bar for done-check
    n_val_bars    = val_end_idx - val_start_idx

    logger.info(
        "Validation window: %s → %s  (%d bars)",
        df_1h.index[val_start_idx].date(),
        df_1h.index[val_end_idx].date(),
        n_val_bars,
    )

    # ── Load agent ─────────────────────────────────────────────────────────────
    env     = AdvancedTradingEnv(df_1h, df_4h, df_1d)
    agent   = DQNAgent(state_dim=env.obs_dim, action_dim=ACTION_DIM)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    agent.load(model_path)
    logger.info("Loaded model from %s  (episode %d, ε=%.4f)",
                model_path, agent.episode, agent.epsilon)

    # ── Run greedy episode ─────────────────────────────────────────────────────
    obs = env.reset(start_step=val_start_idx, max_steps=n_val_bars)

    equity_curve  = [cfg.INITIAL_BALANCE]
    step_indices  = [val_start_idx]

    while True:
        action             = agent.select_action(obs, greedy=True)
        obs, _, done, info = env.step(action)
        equity_curve.append(info["equity"])
        step_indices.append(env.current_step)
        if done:
            break

    trades  = env.trade_log
    summary = env.trade_summary()

    # ── Metrics ────────────────────────────────────────────────────────────────
    eq  = np.array(equity_curve, dtype=float)
    ret = np.diff(eq) / (eq[:-1] + 1e-9)

    # Max drawdown
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / (peak + 1e-9)
    max_dd  = float(dd.min())

    # Annualised Sharpe (8 760 hourly periods per year)
    sharpe = (
        float(ret.mean() / ret.std() * np.sqrt(8_760))
        if len(ret) > 1 and ret.std() > 1e-12
        else 0.0
    )

    pf     = summary.get("profit_factor", 0.0)
    avg_w  = summary.get("avg_win",  0.0)
    avg_l  = summary.get("avg_loss", 0.0)
    avg_rr = abs(avg_w / (avg_l - 1e-9)) if avg_l < 0 else 0.0

    # Monthly PnL breakdown
    monthly: dict = {}
    for t in trades:
        idx = min(t["step"], len(df_1h) - 1)
        key = df_1h.index[idx].strftime("%Y-%m")
        monthly[key] = monthly.get(key, 0.0) + t["pnl"]

    # ── Print report ───────────────────────────────────────────────────────────
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("  BACKTEST REPORT — 2023-2025 OUT-OF-SAMPLE")
    print(SEP)
    print(f"  Model          : {model_path}")
    print(f"  Period         : {df_1h.index[val_start_idx].date()} → "
          f"{df_1h.index[val_end_idx].date()}")
    print(f"  Bars           : {n_val_bars:,}")
    print()
    print(f"  Total Trades   : {summary.get('total_trades', 0)}")
    print(f"  Win Rate       : {summary.get('win_rate', 0):.1%}")
    print(f"  TP / SL        : {summary.get('tp_count', 0)} / "
          f"{summary.get('sl_count', 0)}")
    print(f"  Total PnL      : ${summary.get('total_pnl', 0):+,.2f}")
    print(f"  Final Balance  : ${summary.get('final_balance', cfg.INITIAL_BALANCE):,.2f}")
    print()
    print(f"  Max Drawdown   : {max_dd:.2%}")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Profit Factor  : {pf:.3f}")
    print(f"  Avg R:R        : {avg_rr:.2f}")
    print()

    if monthly:
        max_abs = max(abs(v) for v in monthly.values()) or 1.0
        print("  Monthly PnL Breakdown:")
        for month in sorted(monthly):
            pnl  = monthly[month]
            bar  = "█" * max(1, int(abs(pnl) / max_abs * 20))
            sign = "▲" if pnl >= 0 else "▼"
            print(f"    {month}  {sign}  ${pnl:+9,.2f}  {bar}")

    print(SEP)

    # ── Save equity curve chart ────────────────────────────────────────────────
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    timestamps = [df_1h.index[min(i, len(df_1h) - 1)] for i in step_indices]
    fig, axes  = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1 = axes[0]
    ax1.plot(timestamps, eq, color="#2196F3", linewidth=0.9, label="Equity")
    ax1.fill_between(timestamps, cfg.INITIAL_BALANCE, eq,
                     where=(eq >= cfg.INITIAL_BALANCE),
                     alpha=0.18, color="#4CAF50", label="Profit")
    ax1.fill_between(timestamps, cfg.INITIAL_BALANCE, eq,
                     where=(eq < cfg.INITIAL_BALANCE),
                     alpha=0.18, color="#F44336", label="Loss")
    ax1.axhline(cfg.INITIAL_BALANCE, color="gray", linestyle="--",
                linewidth=0.8, label=f"Start ${cfg.INITIAL_BALANCE:,}")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(
        f"XAUUSD Advanced Agent — Out-of-Sample Backtest 2023-2025\n"
        f"Trades: {summary.get('total_trades', 0)}  |  "
        f"Win Rate: {summary.get('win_rate', 0):.1%}  |  "
        f"Sharpe: {sharpe:.2f}  |  "
        f"Max DD: {max_dd:.2%}  |  "
        f"PF: {pf:.2f}  |  "
        f"Avg R:R: {avg_rr:.2f}",
        fontsize=10,
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.fill_between(timestamps, 0, dd * 100, color="#F44336", alpha=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Equity curve saved → %s", CHART_PATH)
    print(f"\n  Chart saved → {CHART_PATH}\n")

    return {
        "total_trades":  summary.get("total_trades", 0),
        "win_rate":      summary.get("win_rate", 0),
        "total_pnl":     summary.get("total_pnl", 0),
        "final_balance": summary.get("final_balance", cfg.INITIAL_BALANCE),
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "profit_factor": pf,
        "avg_rr":        avg_rr,
        "monthly":       monthly,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run out-of-sample backtest on 2023-2025 data."
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model file (default: best model from config)",
    )
    args = parser.parse_args()
    run_backtest(args.model)
