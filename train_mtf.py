"""
Multi-Timeframe DQN Training – 500 episodes
============================================
Uses 1H (entry), 4H (direction), and 1D (trend) data together.

Each episode randomly samples a contiguous MTF_EPISODE_STEPS window from
the training portion of the 1H data.  This provides data augmentation and
ensures the agent sees all market regimes across the 500 episodes.

Usage
-----
  python train_mtf.py
  python train_mtf.py --episodes 200   # shorter run
  python train_mtf.py --resume         # continue from checkpoint
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_mtf")

# ── Config overrides for MTF run ──────────────────────────────────────────────
import src.config as cfg

cfg.HIDDEN_DIM        = cfg.MTF_HIDDEN_DIM    # 128
cfg.BATCH_SIZE        = cfg.MTF_BATCH_SIZE    # 64
cfg.REPLAY_BUFFER_SIZE= cfg.MTF_BUFFER_SIZE   # 100_000
cfg.MIN_REPLAY_SIZE   = cfg.MTF_MIN_REPLAY    # 2_000
cfg.LEARN_EVERY       = cfg.MTF_LEARN_EVERY   # 16
cfg.EPSILON_DECAY     = 0.9990                # decay over ~460 eps to 0.05
cfg.TARGET_UPDATE_FREQ= 10

from src.data import load_historical_csv, add_indicators
from src.agent import DQNAgent
from src.environment_mtf import MultiTimeframeEnv
from src.train import run_episode, evaluate   # reuse existing helpers

# ── Load & prepare data ───────────────────────────────────────────────────────
def load_and_prep(path):
    df = load_historical_csv(path)
    df = add_indicators(df)
    df.dropna(inplace=True)
    return df

logger.info("Loading 1H data …")
df_1h = load_and_prep(cfg.MTF_DATA_1H)
logger.info("Loading 4H data …")
df_4h = load_and_prep(cfg.MTF_DATA_4H)
logger.info("Loading 1D data …")
df_1d = load_and_prep(cfg.MTF_DATA_1D)

logger.info("1H: %d candles  |  4H: %d candles  |  1D: %d candles",
            len(df_1h), len(df_4h), len(df_1d))

# ── Train / test split (on 1H index) ─────────────────────────────────────────
n_1h       = len(df_1h)
train_end  = int(n_1h * cfg.TRAIN_SPLIT)      # ~80 % of 1H bars
# test set: fixed window at the end of the dataset
test_start = train_end
test_end   = n_1h - 1

logger.info("Train pool: bars 0–%d  |  Test: bars %d–%d",
            train_end, test_start, test_end)

# Build a fixed test environment that always starts at test_start
env_full   = MultiTimeframeEnv(df_1h, df_4h, df_1d)
MIN_START  = env_full.min_step          # ~121 bars
EPISODE_LEN = cfg.MTF_EPISODE_STEPS     # 1500

# Sanity: enough training data?
assert train_end - MIN_START > EPISODE_LEN, \
    "Not enough training data for one episode."

# ── Agent ─────────────────────────────────────────────────────────────────────
obs_dim = env_full.obs_dim              # 385 = 35 candles × 11 features
agent   = DQNAgent(state_dim=obs_dim)
logger.info("Agent  |  state_dim=%d  |  device=%s", obs_dim, agent.device)

# ── Resume ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--resume",   action="store_true")
args = parser.parse_args()

if args.resume and os.path.exists(cfg.MTF_CHECKPOINT_PATH):
    agent.load(cfg.MTF_CHECKPOINT_PATH)
    logger.info("Resumed from checkpoint (episode %d)", agent.episode)

# ── Benchmark first episode ───────────────────────────────────────────────────
logger.info("Benchmarking one episode …")
rng = np.random.default_rng(0)
s   = int(rng.integers(MIN_START, train_end - EPISODE_LEN))
obs = env_full.reset(start_step=s, max_steps=EPISODE_LEN)
t0  = time.time()
while True:
    action              = agent.select_action(obs)
    obs, _, done, _     = env_full.step(action)
    agent.remember(*([obs] * 2 + [0, obs, done]))   # dummy remember for timing
    if done:
        break
bench_s = time.time() - t0
logger.info("1 episode ≈ %.1f s  →  %d episodes ≈ %.0f min",
            bench_s, args.episodes, bench_s * args.episodes / 60)

# ── Training loop ─────────────────────────────────────────────────────────────
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,   exist_ok=True)

best_test_pnl = float("-inf")
rng           = np.random.default_rng()    # fresh RNG

with open(cfg.MTF_LOG_PATH, "w") as flog:
    flog.write("ep,epsilon,train_reward,train_pnl,train_wr,train_trades,"
               "test_pnl,test_wr,avg_loss,elapsed\n")

logger.info("\n%s\n  MTF DQN TRAINING  –  %d episodes\n%s",
            "=" * 60, args.episodes, "=" * 60)

for ep in range(1, args.episodes + 1):
    t0 = time.time()

    # Random window from training pool
    max_start = train_end - EPISODE_LEN
    start     = int(rng.integers(MIN_START, max_start))
    obs       = env_full.reset(start_step=start, max_steps=EPISODE_LEN)

    # ── Run one training episode ──────────────────────────────────────────
    total_reward = 0.0
    losses       = []
    step_count   = 0

    while True:
        action                         = agent.select_action(obs)
        next_obs, reward, done, _      = env_full.step(action)
        agent.remember(obs, action, reward, next_obs, done)

        if step_count % cfg.LEARN_EVERY == 0:
            loss = agent.learn()
            if loss:
                losses.append(loss)

        obs           = next_obs
        total_reward += reward
        step_count   += 1
        if done:
            break

    train_stats = env_full.trade_summary()
    agent.update_epsilon()
    if ep % cfg.TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # ── Evaluation on test set ────────────────────────────────────────────
    test_stats = {}
    if ep % cfg.MTF_EVAL_FREQ == 0 or ep == args.episodes:
        test_steps = test_end - test_start
        env_full.reset(start_step=test_start, max_steps=test_steps)
        obs_t = env_full.reset(start_step=test_start, max_steps=test_steps)
        t_reward = 0.0
        while True:
            a                         = agent.select_action(obs_t, greedy=True)
            obs_t, r, done_t, _       = env_full.step(a)
            t_reward                 += r
            if done_t:
                break
        test_stats = env_full.trade_summary()

        if test_stats.get("total_pnl", 0) > best_test_pnl:
            best_test_pnl = test_stats["total_pnl"]
            agent.save(cfg.MTF_BEST_MODEL_PATH)
            logger.info("★ New best  test_pnl=%.2f  (ep %d)", best_test_pnl, ep)

    if ep % cfg.MTF_SAVE_FREQ == 0:
        agent.save(cfg.MTF_CHECKPOINT_PATH)

    elapsed = time.time() - t0
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    logger.info(
        "Ep %4d/%d | ε=%.4f | "
        "train[r=%6.2f pnl=%8.1f wr=%.2f t=%4d] | "
        "test_pnl=%8.1f | loss=%8.4f | %.1fs",
        ep, args.episodes,
        agent.epsilon,
        total_reward,
        train_stats.get("total_pnl",   0),
        train_stats.get("win_rate",    0),
        train_stats.get("total_trades",0),
        test_stats.get("total_pnl", float("nan")),
        avg_loss,
        elapsed,
    )

    with open(cfg.MTF_LOG_PATH, "a") as flog:
        flog.write(f"{ep},{agent.epsilon:.5f},{total_reward:.4f},"
                   f"{train_stats.get('total_pnl',0):.2f},"
                   f"{train_stats.get('win_rate',0):.4f},"
                   f"{train_stats.get('total_trades',0)},"
                   f"{test_stats.get('total_pnl','')},"
                   f"{test_stats.get('win_rate','')},"
                   f"{avg_loss:.6f},{elapsed:.1f}\n")

# ── Final save & report ───────────────────────────────────────────────────────
agent.save(cfg.MTF_CHECKPOINT_PATH)

print("\n" + "=" * 62)
print("  MTF TRAINING COMPLETE")
print("=" * 62)
print(f"  Episodes         : {args.episodes}")
print(f"  Gradient steps   : {agent.train_steps:,}")
print(f"  Final epsilon    : {agent.epsilon:.4f}")
print(f"  Best test PnL    : ${best_test_pnl:+.2f}")
print(f"  Best model saved : {cfg.MTF_BEST_MODEL_PATH}")
print(f"  Training log     : {cfg.MTF_LOG_PATH}")
print("=" * 62)
