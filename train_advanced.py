"""
V2 Advanced MTF Training — CNN-LSTM DDQN + PER
================================================
Trains the hybrid CNN-LSTM agent with:
  • CNN layers: discover price patterns (support/resistance, breakouts)
  • LSTM layers: learn temporal sequences
  • Double DQN + Prioritized Experience Replay
  • Dynamic ATR-based SL/TP (adapts to current volatility)
  • Improved reward shaping (R:R bonus, overtrade penalty, quick-SL penalty)
  • All original rules: limit/stop orders only, 1% risk, multiple orders

Data split (date-based, no leakage)
  • Train   : 2004 – 2022  (all 1H bars before 2023-01-01)
  • Validate: 2023 – 2025  (completely unseen)

Early stopping
  • Stops if validation PnL does not improve for 100 consecutive episodes.

Usage
-----
  python train_advanced.py                   # fresh 2000-episode run
  python train_advanced.py --episodes 500    # shorter run
  python train_advanced.py --resume          # continue from checkpoint
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
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

import src.config as cfg
from src.data import load_historical_csv, add_indicators, add_v2_features
from src.environment_v2 import TradingEnvV2, V2_ACTION_DIM
from src.agent_cnn_lstm import CNNLSTMAgent

# ── Load + prepare data ───────────────────────────────────────────────────────

def load_tf(path: str) -> pd.DataFrame:
    df = load_historical_csv(path)
    df = add_indicators(df)
    df = add_v2_features(df)
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

# ── Date-based train / validate split ─────────────────────────────────────────
SPLIT_DATE = pd.Timestamp("2023-01-01")
train_end  = int((df_1h.index < SPLIT_DATE).sum())
val_start  = train_end
n_1h       = len(df_1h)

logger.info("Train : 1H bars 0 … %d  (%s → %s)",
            train_end - 1, df_1h.index[0].date(),
            df_1h.index[train_end - 1].date())
logger.info("Val   : 1H bars %d … %d  (%s → %s)",
            val_start, n_1h - 1,
            df_1h.index[val_start].date(), df_1h.index[-1].date())

if not (df_1h.index >= SPLIT_DATE).any():
    raise ValueError("No validation data after 2023-01-01.")

# ── Environment ───────────────────────────────────────────────────────────────
env = TradingEnvV2(df_1h, df_4h, df_1d)

MIN_START = env.min_step
EP_LEN    = cfg.V2_EPISODE_STEPS   # 1500

assert train_end - MIN_START > EP_LEN, (
    f"Not enough training data (train_end={train_end}, "
    f"min_step={MIN_START}, EP_LEN={EP_LEN})."
)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--resume",   action="store_true")
args = parser.parse_args()

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = CNNLSTMAgent(
    action_dim      = V2_ACTION_DIM,
    hidden_dim      = cfg.V2_HIDDEN_DIM,
    lr              = cfg.LEARNING_RATE,
    gamma           = cfg.GAMMA,
    epsilon_decay   = cfg.V2_EPSILON_DECAY,
    buffer_capacity = cfg.V2_BUFFER_SIZE,
    batch_size      = cfg.V2_BATCH_SIZE,
    min_replay      = cfg.V2_MIN_REPLAY,
    per_alpha       = cfg.V2_PER_ALPHA,
    per_beta_start  = cfg.V2_PER_BETA_START,
    per_beta_steps  = cfg.V2_PER_BETA_STEPS,
)
logger.info("Agent | obs_dim=%d | action_dim=%d | device=%s",
            agent.obs_dim, V2_ACTION_DIM, agent.device)

if args.resume and os.path.exists(cfg.V2_CHECKPOINT_PATH):
    agent.load(cfg.V2_CHECKPOINT_PATH)
    logger.info("Resumed from checkpoint (ep %d, ε=%.4f)",
                agent.episode, agent.epsilon)

# ── Quick benchmark ───────────────────────────────────────────────────────────
logger.info("Benchmarking 1 episode …")
rng = np.random.default_rng(0)
s   = int(rng.integers(MIN_START, train_end - EP_LEN))
obs = env.reset(start_step=s, max_steps=EP_LEN)
t0  = time.time()
steps_bench = 0
while True:
    a               = agent.select_action(obs)
    obs, _, done, _ = env.step(a)
    if done:
        break
    steps_bench += 1
bench = time.time() - t0
logger.info("Env: %.1f s / %d steps → est. %.0f min for %d eps (excl. learning)",
            bench, steps_bench, bench * args.episodes / 60, args.episodes)

# ── Output files ──────────────────────────────────────────────────────────────
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR,   exist_ok=True)

with open(cfg.V2_LOG_PATH, "w") as f:
    f.write("ep,epsilon,train_reward,train_pnl,train_wr,train_trades,"
            "train_tp,train_sl,val_pnl,val_wr,val_trades,"
            "val_tp,val_sl,avg_loss,elapsed\n")

# ── CSV helper (defined before the loop that uses it) ─────────────────────────
def _csv_row(ep, agent, train_reward, train_s, val_s, avg_loss, elapsed) -> str:
    return (
        f"{ep},{agent.epsilon:.5f},{train_reward:.4f},"
        f"{train_s.get('total_pnl',0):.2f},"
        f"{train_s.get('win_rate',0):.4f},"
        f"{train_s.get('total_trades',0)},"
        f"{train_s.get('tp_count',0)},"
        f"{train_s.get('sl_count',0)},"
        f"{val_s.get('total_pnl','')},"
        f"{val_s.get('win_rate','')},"
        f"{val_s.get('total_trades','')},"
        f"{val_s.get('tp_count','')},"
        f"{val_s.get('sl_count','')},"
        f"{avg_loss:.6f},{elapsed:.1f}\n"
    )


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_pnl   = float("-inf")
last_best_ep   = agent.episode
early_stop_pat = 100
rng            = np.random.default_rng()
start_ep       = agent.episode + 1

logger.info("\n%s\n  V2 CNN-LSTM AGENT — %d EPISODES\n%s",
            "=" * 64, args.episodes, "=" * 64)
logger.info("Architecture : CNN(3→5→3) → LSTM(2-layer) × 3 branches → Dueling DQN")
logger.info("Learning     : Double DQN + PER (α=%.1f β₀=%.1f)",
            cfg.V2_PER_ALPHA, cfg.V2_PER_BETA_START)
logger.info("SL/TP        : dynamic ATR × {%.1f / %.1f or %.1f}",
            cfg.V2_SL_ATR_MULT, *cfg.V2_TP_ATR_MULTS)
logger.info("Train split  : 2004–2022 (%d bars) | Val: 2023–2025 (%d bars)",
            train_end, n_1h - val_start)
logger.info("Early stop   : patience %d episodes", early_stop_pat)
logger.info("%s", "=" * 64)

for ep in range(start_ep, start_ep + args.episodes):
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

        if step_n % cfg.V2_LEARN_EVERY == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        obs           = next_obs
        total_reward += r
        step_n       += 1
        if done:
            break

    train_s = env.trade_summary()
    agent.update_epsilon()

    # ── Validation ────────────────────────────────────────────────────────────
    val_s = {}
    do_eval = (ep % cfg.V2_EVAL_FREQ == 0) or (ep == start_ep + args.episodes - 1)
    if do_eval:
        val_steps = (n_1h - 1) - val_start
        obs_v = env.reset(start_step=val_start, max_steps=val_steps)
        while True:
            a                    = agent.select_action(obs_v, greedy=True)
            obs_v, _, done_v, _  = env.step(a)
            if done_v:
                break
        val_s = env.trade_summary()

        val_pnl = val_s.get("total_pnl", float("-inf"))
        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            last_best_ep = ep
            agent.save(cfg.V2_BEST_MODEL_PATH)
            logger.info("★ New best  val_pnl=$%.2f  (ep %d)", best_val_pnl, ep)

        # Early stopping check
        if ep - last_best_ep >= early_stop_pat:
            logger.info(
                "Early stopping: no val improvement for %d episodes "
                "(last best ep %d, $%.2f).",
                ep - last_best_ep, last_best_ep, best_val_pnl,
            )
            agent.save(cfg.V2_CHECKPOINT_PATH)
            elapsed  = time.time() - t0
            avg_loss = float(np.mean(losses)) if losses else float("nan")
            with open(cfg.V2_LOG_PATH, "a") as f:
                f.write(_csv_row(ep, agent, total_reward, train_s, val_s, avg_loss, elapsed))
            break

    if ep % cfg.V2_SAVE_FREQ == 0:
        agent.save(cfg.V2_CHECKPOINT_PATH)

    elapsed  = time.time() - t0
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    logger.info(
        "Ep %4d/%d | ε=%.4f | β=%.3f | "
        "train[r=%6.2f pnl=%8.1f wr=%.2f t=%3d TP=%2d SL=%2d] | "
        "val_pnl=%8.1f | loss=%7.4f | %.1fs",
        ep, start_ep + args.episodes - 1,
        agent.epsilon,
        agent.buffer.beta,
        total_reward,
        train_s.get("total_pnl",    0),
        train_s.get("win_rate",     0),
        train_s.get("total_trades", 0),
        train_s.get("tp_count",     0),
        train_s.get("sl_count",     0),
        val_s.get("total_pnl", float("nan")),
        avg_loss,
        elapsed,
    )

    with open(cfg.V2_LOG_PATH, "a") as f:
        f.write(_csv_row(ep, agent, total_reward, train_s, val_s, avg_loss, elapsed))


# ── Final save ────────────────────────────────────────────────────────────────
agent.save(cfg.V2_CHECKPOINT_PATH)

print("\n" + "=" * 64)
print("  V2 CNN-LSTM TRAINING COMPLETE")
print("=" * 64)
print(f"  Episodes run     : {ep - start_ep + 1}")
print(f"  Gradient steps   : {agent.train_steps:,}")
print(f"  Final epsilon    : {agent.epsilon:.4f}")
print(f"  Best val PnL     : ${best_val_pnl:+.2f}")
print(f"  Checkpoint saved : {cfg.V2_CHECKPOINT_PATH}")
print(f"  Best model saved : {cfg.V2_BEST_MODEL_PATH}")
print(f"  Training log     : {cfg.V2_LOG_PATH}")
print("=" * 64)

# ── Run backtest on the best model ────────────────────────────────────────────
print("\nRunning backtest report on best model …")
try:
    from backtest_report import run_backtest
    run_backtest(cfg.V2_BEST_MODEL_PATH)
except Exception as exc:
    logger.warning("Backtest report failed: %s", exc)
