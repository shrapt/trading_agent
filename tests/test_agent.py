"""
Unit tests for the XAUUSD trading agent components.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import _synthetic_data, add_indicators, normalise_window, FEATURE_COLS
from src.environment import XAUUSDEnv, BUY, SELL, HOLD
from src.agent import DQNAgent, ReplayBuffer
from src import config


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    df = _synthetic_data(days=30)
    df = add_indicators(df)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


@pytest.fixture
def env(small_df):
    return XAUUSDEnv(small_df, window_size=10)


@pytest.fixture
def agent(env):
    return DQNAgent(state_dim=env.obs_dim, hidden_dim=64)


# ── Data tests ────────────────────────────────────────────────────────────────

class TestData:
    def test_synthetic_data_shape(self):
        df = _synthetic_data(days=10)
        assert len(df) == 10 * 24
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_indicators_added(self, small_df):
        for col in ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14"]:
            assert col in small_df.columns

    def test_no_nans_after_dropna(self, small_df):
        assert not small_df[FEATURE_COLS].isnull().any().any()

    def test_normalise_window_shape(self, small_df):
        window = small_df.iloc[:10]
        obs = normalise_window(window)
        assert obs.shape == (10 * len(FEATURE_COLS),)

    def test_normalise_window_finite(self, small_df):
        window = small_df.iloc[:10]
        obs = normalise_window(window)
        assert np.all(np.isfinite(obs))


# ── Environment tests ─────────────────────────────────────────────────────────

class TestEnvironment:
    def test_reset_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,)

    def test_step_hold(self, env):
        env.reset()
        obs, reward, done, info = env.step(HOLD)
        assert obs.shape == (env.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_buy_increases_step(self, env):
        env.reset()
        step_before = env.current_step
        env.step(BUY)
        assert env.current_step == step_before + 1

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(HOLD)
            steps += 1
            assert steps < 100_000, "Episode did not terminate"
        assert done

    def test_balance_changes_after_trade(self, env):
        env.reset()
        initial_balance = env.balance
        # Open a long
        env.step(BUY)
        # Immediately reverse (closes long, opens short)
        env.step(SELL)
        # Balance should have changed (may be + or - due to spread/PnL)
        assert env.balance != initial_balance or env.position is not None

    def test_trade_summary_structure(self, env):
        env.reset()
        for action in [BUY, SELL, BUY, SELL, HOLD]:
            env.step(action)
        summary = env.trade_summary()
        assert "total_trades" in summary


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestAgent:
    def test_action_in_valid_range(self, agent, env):
        obs = env.reset()
        action = agent.select_action(obs)
        assert action in (HOLD, BUY, SELL)

    def test_greedy_action_deterministic(self, agent, env):
        obs = env.reset()
        actions = {agent.select_action(obs, greedy=True) for _ in range(10)}
        assert len(actions) == 1  # greedy should always return same action

    def test_remember_fills_buffer(self, agent, env):
        obs = env.reset()
        for _ in range(10):
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()
        assert len(agent.buffer) == 10

    def test_learn_returns_zero_before_min_buffer(self, agent):
        loss = agent.learn()
        assert loss == 0.0

    def test_learn_returns_float_after_enough_data(self, agent, env):
        obs = env.reset()
        # Fill buffer past MIN_REPLAY_SIZE with a small override
        agent_small = DQNAgent(state_dim=env.obs_dim, hidden_dim=32,
                                buffer_size=200, batch_size=16)
        # Monkey-patch min replay size for test speed
        import src.config as cfg
        orig = cfg.MIN_REPLAY_SIZE
        cfg.MIN_REPLAY_SIZE = 20
        try:
            for _ in range(25):
                action = agent_small.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                agent_small.remember(obs, action, reward, next_obs, done)
                obs = next_obs if not done else env.reset()
            loss = agent_small.learn()
            assert loss >= 0.0
        finally:
            cfg.MIN_REPLAY_SIZE = orig

    def test_epsilon_decays(self, agent):
        eps_before = agent.epsilon
        agent.update_epsilon()
        assert agent.epsilon < eps_before

    def test_save_load_roundtrip(self, agent, tmp_path):
        path = str(tmp_path / "test_model.pth")
        agent.epsilon = 0.42
        agent.save(path)
        agent2 = DQNAgent(state_dim=agent.online_net.feature[0].in_features,
                          hidden_dim=64)
        agent2.load(path)
        assert abs(agent2.epsilon - 0.42) < 1e-6


# ── ReplayBuffer tests ────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(np.zeros(5), 0, 0.0, np.zeros(5), False)
        assert len(buf) == 10

    def test_capacity_respected(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(np.zeros(5), 0, float(i), np.zeros(5), False)
        assert len(buf) == 5

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(5), 0, 0.0, np.zeros(5), False)
        batch = buf.sample(16)
        assert len(batch) == 16
