"""
Training loop for the XAUUSD DQN Trading Agent
===============================================

Usage
-----
  python -m src.train                    # full training run
  python -m src.train --episodes 100     # quick test
  python -m src.train --resume           # continue from last checkpoint
"""

import argparse
import logging
import os
import time
from typing import List

import numpy as np

from src import config
from src.data import fetch_data
from src.environment import XAUUSDEnv
from src.agent import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_envs(df):
    """Split dataframe into train / test environments."""
    split_idx = int(len(df) * config.TRAIN_SPLIT)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    test_df   = df.iloc[split_idx:].reset_index(drop=True)
    return XAUUSDEnv(train_df), XAUUSDEnv(test_df)


def run_episode(env: XAUUSDEnv, agent: DQNAgent,
                train: bool = True) -> dict:
    """
    Run one complete episode.
    If `train=True`, the agent learns from each step.
    """
    state  = env.reset()
    done   = False
    total_reward = 0.0
    losses: List[float] = []
    steps  = 0

    while not done:
        action = agent.select_action(state, greedy=not train)
        next_state, reward, done, _ = env.step(action)

        if train:
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss:
                losses.append(loss)

        state         = next_state
        total_reward += reward
        steps        += 1

    summary = env.trade_summary()
    return {
        "total_reward": total_reward,
        "steps":        steps,
        "avg_loss":     float(np.mean(losses)) if losses else 0.0,
        **summary,
    }


def evaluate(env: XAUUSDEnv, agent: DQNAgent) -> dict:
    """Run one greedy evaluation episode and return stats."""
    return run_episode(env, agent, train=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(num_episodes: int = config.NUM_EPISODES, resume: bool = False):
    logger.info("=" * 60)
    logger.info("XAUUSD DQN Trading Agent – Training")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = fetch_data(
        symbol       = config.SYMBOL,
        interval     = config.INTERVAL,
        lookback_days= config.LOOKBACK_DAYS,
        cache_path   = config.DATA_CACHE_PATH,
    )
    train_env, test_env = make_envs(df)
    logger.info("Train env: %d candles | Test env: %d candles",
                len(train_env.df), len(test_env.df))

    # ── Agent ─────────────────────────────────────────────────────────────────
    state_dim = train_env.obs_dim
    agent = DQNAgent(state_dim=state_dim)

    if resume:
        agent.load(config.CHECKPOINT_PATH)

    # ── Metrics ───────────────────────────────────────────────────────────────
    best_test_pnl   = float("-inf")
    history         = []

    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "training.csv")
    with open(log_path, "w") as f:
        f.write("episode,epsilon,train_reward,train_pnl,train_win_rate,"
                "test_reward,test_pnl,test_win_rate,avg_loss\n")

    # ── Loop ──────────────────────────────────────────────────────────────────
    for ep in range(1, num_episodes + 1):
        t0     = time.time()
        stats  = run_episode(train_env, agent, train=True)

        agent.update_epsilon()

        if ep % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # ── Evaluation ────────────────────────────────────────────────────────
        test_stats = {}
        if ep % config.EVAL_FREQ == 0 or ep == num_episodes:
            test_stats = evaluate(test_env, agent)

            if test_stats.get("total_pnl", 0) > best_test_pnl:
                best_test_pnl = test_stats["total_pnl"]
                agent.save(config.BEST_MODEL_PATH)
                logger.info("★ New best model saved  (test PnL=%.2f)", best_test_pnl)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if ep % config.SAVE_FREQ == 0:
            agent.save(config.CHECKPOINT_PATH)

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        logger.info(
            "Ep %4d/%d | ε=%.4f | "
            "train[r=%.3f pnl=%.1f wr=%.2f trades=%d] | "
            "test[pnl=%.1f wr=%.2f] | loss=%.5f | %.1fs",
            ep, num_episodes,
            agent.epsilon,
            stats["total_reward"],
            stats.get("total_pnl", 0),
            stats.get("win_rate",  0),
            stats.get("total_trades", 0),
            test_stats.get("total_pnl", float("nan")),
            test_stats.get("win_rate",  float("nan")),
            stats["avg_loss"],
            elapsed,
        )

        with open(log_path, "a") as f:
            f.write(f"{ep},{agent.epsilon:.5f},"
                    f"{stats['total_reward']:.4f},"
                    f"{stats.get('total_pnl', 0):.2f},"
                    f"{stats.get('win_rate', 0):.4f},"
                    f"{test_stats.get('total_reward', '')},{test_stats.get('total_pnl', '')},"
                    f"{test_stats.get('win_rate', '')},"
                    f"{stats['avg_loss']:.6f}\n")

        history.append(stats)

    # ── Final save ────────────────────────────────────────────────────────────
    agent.save(config.CHECKPOINT_PATH)
    logger.info("Training complete. Best test PnL: %.2f", best_test_pnl)
    logger.info("Training log: %s", log_path)
    return agent, history


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XAUUSD DQN agent")
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()
    train(num_episodes=args.episodes, resume=args.resume)
