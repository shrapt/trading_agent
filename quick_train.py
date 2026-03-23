"""
Quick training run for a fast demo report.
Uses a small dataset (60 days) and 30 episodes for speed.
"""

import sys, os
sys.path.insert(0, '/home/user/trading_agent')
os.chdir('/home/user/trading_agent')

import logging
import numpy as np
import time
import torch
torch.set_num_threads(4)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("quick_train")

# ── Override config for speed ─────────────────────────────────────────────────
import src.config as cfg
cfg.LOOKBACK_DAYS     = 60      # ~1440 hourly candles
cfg.NUM_EPISODES      = 30
cfg.EVAL_FREQ         = 5
cfg.SAVE_FREQ         = 30
cfg.MIN_REPLAY_SIZE   = 200
cfg.BATCH_SIZE        = 32
cfg.REPLAY_BUFFER_SIZE= 5_000
cfg.HIDDEN_DIM        = 64
cfg.LEARN_EVERY       = 32
cfg.DATA_CACHE_PATH   = "data/quick_cache.csv"

from src.data import load_historical_csv, _synthetic_data, add_indicators
from src.environment import XAUUSDEnv
from src.agent import DQNAgent
from src.train import run_episode, evaluate

# ── Data ──────────────────────────────────────────────────────────────────────
hist_path = cfg.HISTORICAL_DATA_PATH
if os.path.exists(hist_path):
    logger.info("Loading real XAUUSD data from '%s' …", hist_path)
    df = load_historical_csv(hist_path)
    df = add_indicators(df)
    df.dropna(inplace=True)
    # Use the most recent 3000 candles (~125 days) for speed
    df = df.iloc[-3000:].reset_index(drop=True)
    logger.info("Using %d candles (most recent)", len(df))
else:
    logger.info("Historical file not found, using synthetic data …")
    df = _synthetic_data(days=60)
    df = add_indicators(df)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
logger.info("Dataset: %d candles", len(df))

split = int(len(df) * 0.80)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)

train_env = XAUUSDEnv(train_df)
test_env  = XAUUSDEnv(test_df)
logger.info("Train: %d candles | Test: %d candles", len(train_df), len(test_df))

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = DQNAgent(state_dim=train_env.obs_dim)
logger.info("Agent initialised on device: %s", agent.device)

# ── Track learning ────────────────────────────────────────────────────────────
history = []
early_episodes  = []   # ep 1-5
middle_episodes = []   # ep 13-17
late_episodes   = []   # ep 26-30

logger.info("\n=== STARTING TRAINING (30 episodes) ===\n")

for ep in range(1, 31):
    t0    = time.time()
    stats = run_episode(train_env, agent, train=True)
    agent.update_epsilon()
    if ep % cfg.TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    test_stats = {}
    if ep % cfg.EVAL_FREQ == 0 or ep == 30:
        test_stats = evaluate(test_env, agent)

    elapsed = time.time() - t0
    logger.info(
        "Ep %2d/30 | ε=%.4f | reward=%7.3f | pnl=%8.1f | wr=%.2f | "
        "trades=%4d | test_pnl=%8.1f | loss=%.5f | %.1fs",
        ep, agent.epsilon,
        stats["total_reward"],
        stats.get("total_pnl", 0),
        stats.get("win_rate", 0),
        stats.get("total_trades", 0),
        test_stats.get("total_pnl", float("nan")),
        stats["avg_loss"],
        elapsed,
    )

    row = {**stats, **{f"test_{k}": v for k,v in test_stats.items()}, "ep": ep}
    history.append(row)

    if ep <= 5:
        early_episodes.append(row)
    if 13 <= ep <= 17:
        middle_episodes.append(row)
    if ep >= 26:
        late_episodes.append(row)

# ── Save model ────────────────────────────────────────────────────────────────
agent.save(cfg.CHECKPOINT_PATH)
agent.save(cfg.BEST_MODEL_PATH)

# ── Final eval ────────────────────────────────────────────────────────────────
final_test = evaluate(test_env, agent)

# ── Print learning report ─────────────────────────────────────────────────────
def avg(lst, key):
    vals = [x.get(key, 0) for x in lst if x.get(key) is not None]
    return np.mean(vals) if vals else 0

print("\n" + "="*62)
print("  LEARNING REPORT – XAUUSD DQN Agent")
print("="*62)

print("\n[PHASE 1] Episodes 1–5  (pure exploration, ε≈1.00)")
print(f"  Avg reward   : {avg(early_episodes,'total_reward'):+.3f}")
print(f"  Avg PnL      : ${avg(early_episodes,'total_pnl'):+.1f}")
print(f"  Avg win rate : {avg(early_episodes,'win_rate'):.1%}")
print(f"  Avg trades   : {avg(early_episodes,'total_trades'):.0f}")

print("\n[PHASE 2] Episodes 13–17  (mixed exploration/exploitation, ε≈0.94)")
print(f"  Avg reward   : {avg(middle_episodes,'total_reward'):+.3f}")
print(f"  Avg PnL      : ${avg(middle_episodes,'total_pnl'):+.1f}")
print(f"  Avg win rate : {avg(middle_episodes,'win_rate'):.1%}")
print(f"  Avg trades   : {avg(middle_episodes,'total_trades'):.0f}")

print("\n[PHASE 3] Episodes 26–30  (mostly exploitation, ε≈0.87)")
print(f"  Avg reward   : {avg(late_episodes,'total_reward'):+.3f}")
print(f"  Avg PnL      : ${avg(late_episodes,'total_pnl'):+.1f}")
print(f"  Avg win rate : {avg(late_episodes,'win_rate'):.1%}")
print(f"  Avg trades   : {avg(late_episodes,'total_trades'):.0f}")

print("\n[FINAL TEST SET PERFORMANCE]")
print(f"  Total trades  : {final_test.get('total_trades', 0)}")
print(f"  Win rate      : {final_test.get('win_rate', 0):.1%}")
print(f"  Profit factor : {final_test.get('profit_factor', 0):.2f}")
print(f"  Total PnL     : ${final_test.get('total_pnl', 0):+.2f}")
print(f"  Final balance : ${final_test.get('final_balance', 10000):,.2f}  (start: $10,000)")
print(f"  Return        : {(final_test.get('final_balance',10000)-10000)/10000:.1%}")
print(f"  Avg win       : ${final_test.get('avg_win', 0):+.2f}")
print(f"  Avg loss      : ${final_test.get('avg_loss', 0):+.2f}")

pnl_improvement = avg(late_episodes,'total_pnl') - avg(early_episodes,'total_pnl')
wr_improvement  = avg(late_episodes,'win_rate')  - avg(early_episodes,'win_rate')

print("\n[WHAT THE AGENT LEARNED]")
print(f"  PnL change early→late  : ${pnl_improvement:+.1f}")
print(f"  Win-rate change        : {wr_improvement:+.1%}")
print(f"  Trade frequency change : {avg(late_episodes,'total_trades')-avg(early_episodes,'total_trades'):+.0f} trades/ep")
print(f"  Final epsilon (ε)      : {agent.epsilon:.4f}")
print(f"  Total gradient steps   : {agent.train_steps:,}")
print("="*62)
