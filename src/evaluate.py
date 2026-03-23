"""
Evaluation & paper-trading runner for the trained XAUUSD agent.

Usage
-----
  python -m src.evaluate                   # evaluate best model on test data
  python -m src.evaluate --live            # run on latest fetched data (paper)
  python -m src.evaluate --model models/xauusd_dqn.pth
"""

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.data import fetch_data
from src.environment import XAUUSDEnv, BUY, SELL, HOLD
from src.agent import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


def evaluate_model(model_path: str, use_test_split: bool = True,
                   verbose: bool = False):
    logger.info("Loading data …")
    df = fetch_data(
        symbol       = config.SYMBOL,
        interval     = config.INTERVAL,
        lookback_days= config.LOOKBACK_DAYS,
        cache_path   = config.DATA_CACHE_PATH,
    )

    if use_test_split:
        split_idx = int(len(df) * config.TRAIN_SPLIT)
        df = df.iloc[split_idx:].reset_index(drop=True)
        logger.info("Evaluating on TEST split: %d candles", len(df))
    else:
        logger.info("Evaluating on FULL dataset: %d candles", len(df))

    env   = XAUUSDEnv(df)
    agent = DQNAgent(state_dim=env.obs_dim)
    agent.load(model_path)

    state  = env.reset()
    done   = False
    step   = 0
    equity_curve = [config.INITIAL_BALANCE]

    while not done:
        action                 = agent.select_action(state, greedy=True)
        next_state, _, done, info = env.step(action)

        if verbose:
            price = env.df.loc[env.current_step - 1, "close"]
            logger.info("Step %5d | Price %8.2f | Action %-4s | Equity %10.2f",
                        step, price, ACTION_NAMES[action], info["equity"])

        equity_curve.append(info["equity"])
        state = next_state
        step += 1

    summary = env.trade_summary()
    _print_summary(summary, equity_curve)
    return summary, equity_curve


def _print_summary(summary: dict, equity_curve: list):
    initial  = config.INITIAL_BALANCE
    final    = equity_curve[-1]
    peak     = max(equity_curve)
    trough   = min(equity_curve)
    drawdown = (peak - trough) / peak * 100 if peak else 0

    returns  = pd.Series(equity_curve).pct_change().dropna()
    sharpe   = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 24)

    print("\n" + "=" * 56)
    print("  XAUUSD DQN Agent – Evaluation Results")
    print("=" * 56)
    print(f"  Total trades    : {summary.get('total_trades', 0)}")
    print(f"  Win rate        : {summary.get('win_rate', 0):.1%}")
    print(f"  Profit factor   : {summary.get('profit_factor', 0):.2f}")
    print(f"  Avg win         : ${summary.get('avg_win', 0):.2f}")
    print(f"  Avg loss        : ${summary.get('avg_loss', 0):.2f}")
    print(f"  Total PnL       : ${summary.get('total_pnl', 0):.2f}")
    print(f"  Final balance   : ${final:.2f}  (start: ${initial:.2f})")
    print(f"  Return          : {(final - initial) / initial:.1%}")
    print(f"  Max drawdown    : {drawdown:.1f}%")
    print(f"  Sharpe ratio    : {sharpe:.2f}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XAUUSD DQN agent")
    parser.add_argument("--model",   default=config.BEST_MODEL_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--full",    action="store_true",
                        help="Use full dataset instead of test split")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every step")
    args = parser.parse_args()

    evaluate_model(
        model_path      = args.model,
        use_test_split  = not args.full,
        verbose         = args.verbose,
    )
