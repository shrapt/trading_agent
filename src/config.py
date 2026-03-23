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
