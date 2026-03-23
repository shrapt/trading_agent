# XAUUSD Self-Learning Trading Agent

A **Deep Q-Network (DQN)** agent that trades Gold (XAUUSD) and learns from its
own mistakes using reinforcement learning.

---

## How the agent learns from mistakes

| Mechanism | What it does |
|---|---|
| **Experience Replay** | Stores every trade outcome in a circular buffer; randomly samples mini-batches to break temporal correlations and re-learn from past mistakes repeatedly. |
| **Negative reward signal** | A losing trade produces a negative reward → the agent lowers its predicted Q-value for that (state, action) pair → it avoids repeating the same mistake in similar market conditions. |
| **Double DQN** | Separates action selection (online network) from action evaluation (target network), preventing overestimation of bad actions. |
| **Dueling architecture** | Separates *state value* V(s) from *action advantage* A(s,a), helping the agent learn when *any* action is bad (e.g., volatile markets). |
| **ε-greedy decay** | Starts with 100 % random exploration, gradually shifting to exploitation of learned strategies as training progresses. |
| **Reward shaping** | Realised PnL + unrealised change + inaction-in-loss penalty → agent learns to cut losses fast and let winners run. |

---

## Architecture

```
State (WINDOW_SIZE × 11 features)
    └─► Linear(256) → LayerNorm → ReLU → Dropout
        └─► Linear(256) → LayerNorm → ReLU → Dropout
            ├─► Value stream    V(s)      → scalar
            └─► Advantage stream A(s,a)  → [HOLD, BUY, SELL]
                └─► Q(s,a) = V(s) + A(s,a) - mean(A)
```

### State features (per candle, last 30 candles)

| Feature | Description |
|---|---|
| open, high, low, close | Normalised OHLC price |
| volume | Normalised volume |
| rsi_14 | RSI momentum [0–1] |
| macd, macd_signal | Trend (relative to price) |
| bb_upper, bb_lower | Bollinger Bands (relative) |
| atr_14 | Volatility (relative) |

### Actions

| ID | Action | Description |
|---|---|---|
| 0 | HOLD | Do nothing / maintain current position |
| 1 | BUY | Open long (closes any existing short first) |
| 2 | SELL | Open short (closes any existing long first) |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (uses synthetic data if yfinance download fails)
python main.py train --episodes 500

# 3. Evaluate the best model on the test split
python main.py eval

# 4. Quick 50-episode demo
python main.py demo
```

### Resume training
```bash
python main.py train --resume
```

### Verbose evaluation (every step printed)
```bash
python main.py eval --verbose
```

---

## Project structure

```
trading_agent/
├── main.py              # CLI entry point
├── requirements.txt
├── src/
│   ├── config.py        # All hyperparameters in one place
│   ├── data.py          # Data fetching, indicators, normalisation
│   ├── environment.py   # Paper-trading gym-style environment
│   ├── agent.py         # DQN agent with experience replay
│   ├── train.py         # Training loop
│   └── evaluate.py      # Evaluation & paper-trading runner
├── tests/
│   └── test_agent.py    # Unit tests (pytest)
├── models/              # Saved checkpoints (created at runtime)
├── data/                # Cached price data (created at runtime)
└── logs/                # Training CSV logs (created at runtime)
```

---

## Configuration (`src/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_SIZE` | 30 | Candles in each observation |
| `INITIAL_BALANCE` | 10 000 | Paper-money starting balance (USD) |
| `LEARNING_RATE` | 1e-4 | Adam optimiser LR |
| `GAMMA` | 0.99 | Reward discount factor |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.05 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.9995 | Per-episode ε multiplier |
| `REPLAY_BUFFER_SIZE` | 50 000 | Experience buffer capacity |
| `BATCH_SIZE` | 64 | Training mini-batch size |
| `TARGET_UPDATE_FREQ` | 10 | Episodes between target-net syncs |
| `NUM_EPISODES` | 500 | Default training episodes |

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Data source

- **Live data:** [yfinance](https://github.com/ranaroussi/yfinance) –
  downloads Gold Futures (`GC=F`) as an XAUUSD proxy.
- **Fallback:** If yfinance is unavailable, the agent trains on realistic
  synthetic price data generated with Geometric Brownian Motion.

> **Disclaimer:** This is a research/educational project. Do **not** use it
> for real money trading without extensive validation.
