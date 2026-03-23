"""
Configuration for XAUUSD Trading Agent
"""

# ── Data ──────────────────────────────────────────────────────────────────────
SYMBOL = "GC=F"          # Yahoo Finance ticker for Gold Futures (XAUUSD proxy)
INTERVAL = "1h"          # Candle interval
LOOKBACK_DAYS = 730      # Days of historical data to fetch for training

# ── Environment ───────────────────────────────────────────────────────────────
WINDOW_SIZE = 30         # Number of candles the agent observes at once
INITIAL_BALANCE = 10_000 # Starting paper-money balance (USD)
LOT_SIZE = 1             # Ounces of gold per trade
SPREAD_PIPS = 0.30       # Simulated spread cost per ounce
MAX_OPEN_TRADES = 1      # Maximum simultaneous positions

# ── Agent / DQN ───────────────────────────────────────────────────────────────
STATE_DIM = WINDOW_SIZE * 10  # features per time-step × window
ACTION_DIM = 3                # 0=HOLD, 1=BUY, 2=SELL
HIDDEN_DIM = 128

LEARNING_RATE = 1e-4
GAMMA = 0.99              # Discount factor
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_END = 0.05        # Minimum exploration rate
EPSILON_DECAY = 0.9995    # Multiplicative decay per episode

REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10   # Update target network every N episodes
MIN_REPLAY_SIZE = 1_000   # Minimum experiences before training starts
LEARN_EVERY = 4           # Run a gradient update every N environment steps

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPISODES = 500
SAVE_FREQ = 50            # Save model checkpoint every N episodes
EVAL_FREQ = 25            # Evaluate on test split every N episodes
TRAIN_SPLIT = 0.80        # 80% train, 20% test

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
DATA_DIR = "data"
LOG_DIR = "logs"
CHECKPOINT_PATH = f"{MODEL_DIR}/xauusd_dqn.pth"
BEST_MODEL_PATH = f"{MODEL_DIR}/xauusd_dqn_best.pth"
DATA_CACHE_PATH = f"{DATA_DIR}/xauusd_cache.csv"
HISTORICAL_DATA_PATH = "historical data"  # real XAUUSD 1h CSV (semicolon-delimited)

# ── Multi-Timeframe (MTF) ─────────────────────────────────────────────────────
MTF_WINDOW_1H    = 20          # hourly candles in observation
MTF_WINDOW_4H    = 10          # 4H candles in observation
MTF_WINDOW_1D    = 5           # daily candles in observation
# state dim = (20+10+5) × 11 features = 385  (computed in environment_mtf.py)
MTF_EPISODE_STEPS = 1500       # 1H candles per training episode (random window)
MTF_HIDDEN_DIM   = 128
MTF_LEARN_EVERY  = 16
MTF_BATCH_SIZE   = 64
MTF_MIN_REPLAY   = 2_000
MTF_BUFFER_SIZE  = 100_000
MTF_EVAL_FREQ    = 25
MTF_SAVE_FREQ    = 50
MTF_DATA_1H = "multi timeframe/1H"
MTF_DATA_4H = "multi timeframe/4H"
MTF_DATA_1D = "multi timeframe/1D"
MTF_CHECKPOINT_PATH = f"{MODEL_DIR}/xauusd_mtf_dqn.pth"
MTF_BEST_MODEL_PATH = f"{MODEL_DIR}/xauusd_mtf_dqn_best.pth"
MTF_LOG_PATH        = f"{LOG_DIR}/mtf_training.csv"

# ── Advanced Trading Agent (limit/stop orders, SL/TP, 1% risk) ────────────────
ADV_ENTRY_OFFSETS  = [0.002, 0.005]    # entry distance from price: 0.2%, 0.5%
ADV_SL_PCTS        = [0.003, 0.006]    # SL distance from entry: 0.3%, 0.6%
ADV_TP_RATIOS      = [1.5, 2.5]        # TP = entry ± SL_dist × ratio
ADV_RISK_PCT       = 0.01              # risk 1% of balance per trade
ADV_MAX_LOT        = 200.0             # max oz per trade (safety cap)
ADV_MAX_PENDING    = 3                 # max simultaneous pending orders
ADV_MAX_OPEN       = 3                 # max simultaneous open positions
ADV_ORDER_EXPIRY   = 24               # cancel pending after N 1H candles
ADV_EPISODE_STEPS  = 1500             # 1H candles per episode
ADV_HIDDEN_DIM     = 256
ADV_LEARN_EVERY    = 64
ADV_BATCH_SIZE     = 64
ADV_MIN_REPLAY     = 2_000
ADV_BUFFER_SIZE    = 100_000
ADV_EPSILON_DECAY  = 0.9940           # reaches 0.05 in 500 episodes
ADV_EVAL_FREQ      = 25
ADV_SAVE_FREQ      = 50
ADV_CHECKPOINT_PATH = f"{MODEL_DIR}/xauusd_adv_dqn.pth"
ADV_BEST_MODEL_PATH = f"{MODEL_DIR}/xauusd_adv_dqn_best.pth"
ADV_LOG_PATH        = f"{LOG_DIR}/advanced_training.csv"

# ── V2: CNN-LSTM DDQN + PER (hybrid self-learning agent) ─────────────────────
# Windows: raw candle counts fed into each CNN-LSTM branch
V2_WINDOW_1H   = 100        # 1H candles per observation
V2_WINDOW_4H   = 50         # 4H candles per observation
V2_WINDOW_1D   = 20         # 1D candles per observation
V2_N_FEAT      = 8          # OHLCV + ATR + price_pos_50 + vol_regime
V2_HIDDEN_DIM  = 256
V2_EPISODE_STEPS = 1500     # 1H bars per training episode (random window)

# Dynamic SL/TP using ATR
V2_SL_ATR_MULT         = 1.5          # SL = 1.5 × ATR
V2_TP_ATR_MULTS        = [2.0, 3.0]   # TP options: 2× or 3× ATR
V2_ENTRY_OFFSETS_ATR   = [0.5, 1.0]   # entry = price ± k × ATR

# Reward shaping knobs
V2_QUICK_SL_BARS       = 3            # SL within this many bars → extra penalty
V2_QUICK_SL_PENALTY    = 0.5          # extra penalty magnitude
V2_OVERTRADE_PENALTY   = 0.02         # trying to place when at max orders
V2_CHOPPY_HOLD_REWARD  = 0.005        # per-step reward for flat book in quiet market
V2_CHOPPY_REGIME_THR   = 0.33         # vol_regime below this → "choppy"

# Agent / training
V2_EPSILON_DECAY   = 0.9980           # ε: 1.0 → 0.05 ≈ 1 500 episodes
V2_BATCH_SIZE      = 64
V2_BUFFER_SIZE     = 100_000
V2_MIN_REPLAY      = 2_000
V2_LEARN_EVERY     = 64
V2_EVAL_FREQ       = 25
V2_SAVE_FREQ       = 50
V2_PER_ALPHA       = 0.6
V2_PER_BETA_START  = 0.4
V2_PER_BETA_STEPS  = 300_000

# Paths
V2_CHECKPOINT_PATH = f"{MODEL_DIR}/xauusd_v2_dqn.pth"
V2_BEST_MODEL_PATH = f"{MODEL_DIR}/xauusd_v2_dqn_best.pth"
V2_LOG_PATH        = f"{LOG_DIR}/v2_training.csv"
