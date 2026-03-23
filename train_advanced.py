"""
Advanced MTF DQN Training — 500 episodes
==========================================
Trains the redesigned agent with:
  • Limit / stop orders only  (no market orders)
  • Immutable SL + TP set at order placement
  • Multiple simultaneous orders (up to 3 pending + 3 open)
  • 1% risk-per-trade position sizing
  • Multi-timeframe state: 1D (trend) + 4H (direction) + 1H (entry)

Each episode randomly samples a contiguous ADV_EPISODE_STEPS window from
the training portion of the 1H dataset, giving diverse market regimes.

Usage
-----
  python train_advanced.py                  # fresh 500-episode run
  python train_advanced.py --episodes 200   # shorter run
  python train_advanced.py --resume         # continue from checkpoint
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_adv")

# ── Config overrides ──────────────────────────────────────────────────────────
import src.config as cfg

cfg.HIDDEN_DIM        = cfg.ADV_HIDDEN_DIM     # 256
cfg.BATCH_SIZE        = cfg.ADV_BATCH_SIZE     # 64
cfg.REPLAY_BUFFER_SIZE= cfg.ADV_BUFFER_SIZE    # 100_000
cfg.MIN_REPLAY_SIZE   = cfg.ADV_MIN_REPLAY     # 2_000
cfg.LEARN_EVERY       = cfg.ADV_LEARN_EVERY    # 64
cfg.EPSILON_DECAY     = cfg.ADV_EPSILON_DECAY  # 0.9940
cfg.EPSILON_END       = 0.05
cfg.TARGET_UPDATE_FREQ= 10
cfg.LEARNING_RATE     = 1e-4
cfg.GAMMA             = 0.99

from src.data import load_historical_csv, add_indicators
from src.environment_advanced import AdvancedTradingEnv, ACTION_DIM
from src.agent import DQNAgent

# ── Load data ─────────────────────────────────────────────────────────────────
def load_tf(path: str):
    df = load_historical_csv(path)
    df = add_indicators(df)
    df.dropna(inplace=True)
    return df

logger.info("Loading 1H data …")
df_1h = load_tf(cfg.MTF_DATA_1H)
logger.info("Loading 4H data …")
df_4h = load_tf(cfg.MTF_DATA_4H)
logger.info("Loading 1D data …")
df_1d = load_tf(cfg.MTF_DATA_1D)

logger.info("1H: %d bars | 4H: %d bars | 1D: %d bars",
            len(df_1h), len(df_4h), len(df_1d))

# ── Train / test split ────────────────────────────────────────────────────────
n_1h       = len(df_1h)
train_end  = int(n_1h * cfg.TRAIN_SPLIT)
test_start = train_end

logger.info("Train pool: 1H bars 0..%d | Test: %d..%d",
            train_end, test_start, n_1h - 1)

# ── Environments ──────────────────────────────────────────────────────────────
env = AdvancedTradingEnv(df_1h, df_4h, df_1d)

MIN_START   = env.min_step
EP_LEN      = cfg.ADV_EPISODE_STEPS   # 1500

assert train_end - MIN_START > EP_LEN, "Not enough training data."

# ── Agent ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--resume",   action="store_true")
args = parser.parse_args()

obs_dim = env.obs_dim        # 393
agent   = DQNAgent(state_dim=obs_dim, action_dim=ACTION_DIM)
logger.info("Agent | state_dim=%d | action_dim=%d | device=%s",
            obs_dim, ACTION_DIM, agent.device)

if args.resume and os.path.exists(cfg.ADV_CHECKPOINT_PATH):
    agent.load(cfg.ADV_CHECKPOINT_PATH)
    logger.info("Resumed from checkpoint (ep %d, ε=%.4f)",
                agent.episode, agent.epsilon)

# ── Benchmark ────────────────────────────────────────────────────────────────
logger.info("Benchmarking 1 episode …")
rng = np.random.default_rng(0)
s   = int(rng.integers(MIN_START, train_end - EP_LEN))
obs = env.reset(start_step=s, max_steps=EP_LEN)
t0  = time.time()
steps = 0
while True:
    a              = agent.select_action(obs)
    obs, _, done, _ = env.step(a)
    if done:
        break
    steps += 1
bench = time.time() - t0
logger.info("Env-only: %.1f s / %d steps  →  est. %.0f min for %d episodes "
            "(excl. learning)",
            bench, steps, bench * args.episodes / 60, args.episodes)

# ── Setup output files ────────────────────────────────────────────────────────
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,   exist_ok=True)

with open(cfg.ADV_LOG_PATH, "w") as f:
    f.write("ep,epsilon,train_reward,train_pnl,train_wr,train_trades,"
            "train_tp,train_sl,test_pnl,test_wr,test_trades,"
            "test_tp,test_sl,avg_loss,elapsed\n")

# ── Training loop ─────────────────────────────────────────────────────────────
best_test_pnl = float("-inf")
rng           = np.random.default_rng()

logger.info("\n%s\n  ADVANCED MTF AGENT — %d EPISODES\n%s",
            "=" * 64, args.episodes, "=" * 64)
logger.info("Order types : limit_buy | limit_sell | stop_buy | stop_sell")
logger.info("SL/TP       : immutable, set at placement")
logger.info("Risk/trade  : 1%% of balance")
logger.info("Max orders  : %d pending + %d open simultaneously",
            cfg.ADV_MAX_PENDING, cfg.ADV_MAX_OPEN)
logger.info("%s", "=" * 64)

for ep in range(1, args.episodes + 1):
    t0 = time.time()

    # Random window from training pool
    max_start = train_end - EP_LEN
    start     = int(rng.integers(MIN_START, max_start))
    obs       = env.reset(start_step=start, max_steps=EP_LEN)

    total_reward = 0.0
    losses       = []
    step_n       = 0

    while True:
        action               = agent.select_action(obs)
        next_obs, r, done, _ = env.step(action)
        agent.remember(obs, action, r, next_obs, done)

        if step_n % cfg.LEARN_EVERY == 0:
            loss = agent.learn()
            if loss:
                losses.append(loss)

        obs           = next_obs
        total_reward += r
        step_n       += 1
        if done:
            break

    train_s = env.trade_summary()
    agent.update_epsilon()
    if ep % cfg.TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # ── Evaluation ───────────────────────────────────────────────────────────
    test_s = {}
    if ep % cfg.ADV_EVAL_FREQ == 0 or ep == args.episodes:
        test_steps = (n_1h - 1) - test_start
        obs_t = env.reset(start_step=test_start, max_steps=test_steps)
        while True:
            a                 = agent.select_action(obs_t, greedy=True)
            obs_t, _, done_t, _ = env.step(a)
            if done_t:
                break
        test_s = env.trade_summary()

        if test_s.get("total_pnl", 0) > best_test_pnl:
            best_test_pnl = test_s["total_pnl"]
            agent.save(cfg.ADV_BEST_MODEL_PATH)
            logger.info("★ New best  test_pnl=$%.2f  (ep %d)",
                        best_test_pnl, ep)

    if ep % cfg.ADV_SAVE_FREQ == 0:
        agent.save(cfg.ADV_CHECKPOINT_PATH)
        logger.info("Checkpoint saved at ep %d", ep)

    elapsed  = time.time() - t0
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    logger.info(
        "Ep %4d/%d | ε=%.4f | "
        "train[r=%6.2f pnl=%8.1f wr=%.2f t=%3d TP=%2d SL=%2d] | "
        "test_pnl=%8.1f | loss=%7.4f | %.1fs",
        ep, args.episodes,
        agent.epsilon,
        total_reward,
        train_s.get("total_pnl",    0),
        train_s.get("win_rate",     0),
        train_s.get("total_trades", 0),
        train_s.get("tp_count",     0),
        train_s.get("sl_count",     0),
        test_s.get("total_pnl", float("nan")),
        avg_loss,
        elapsed,
    )

    with open(cfg.ADV_LOG_PATH, "a") as f:
        f.write(
            f"{ep},{agent.epsilon:.5f},{total_reward:.4f},"
            f"{train_s.get('total_pnl',0):.2f},"
            f"{train_s.get('win_rate',0):.4f},"
            f"{train_s.get('total_trades',0)},"
            f"{train_s.get('tp_count',0)},"
            f"{train_s.get('sl_count',0)},"
            f"{test_s.get('total_pnl','')},"
            f"{test_s.get('win_rate','')},"
            f"{test_s.get('total_trades','')},"
            f"{test_s.get('tp_count','')},"
            f"{test_s.get('sl_count','')},"
            f"{avg_loss:.6f},{elapsed:.1f}\n"
        )

# ── Final save ────────────────────────────────────────────────────────────────
agent.save(cfg.ADV_CHECKPOINT_PATH)

print("\n" + "=" * 64)
print("  ADVANCED MTF TRAINING COMPLETE")
print("=" * 64)
print(f"  Episodes         : {args.episodes}")
print(f"  Gradient steps   : {agent.train_steps:,}")
print(f"  Final epsilon    : {agent.epsilon:.4f}")
print(f"  Best test PnL    : ${best_test_pnl:+.2f}")
print(f"  Checkpoint saved : {cfg.ADV_CHECKPOINT_PATH}")
print(f"  Best model saved : {cfg.ADV_BEST_MODEL_PATH}")
print(f"  Training log     : {cfg.ADV_LOG_PATH}")
print("=" * 64)
