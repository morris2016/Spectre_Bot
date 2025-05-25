#!/usr/bin/env python3
"""Reinforcement Learning Trainer module."""

from typing import Any, Dict, Optional
import pandas as pd

from data_storage.market_data import MarketDataRepository
from feature_service.feature_extraction import FeatureExtractor
from intelligence.adaptive_learning.reinforcement import MarketEnvironment, DQNAgent
from common.async_utils import run_in_threadpool


class RLTradingAgent:
    """Helper class for training and running RL trading agents."""

    def __init__(
        self,
        feature_list: Optional[list] = None,
        market_repo: Optional[MarketDataRepository] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        timeframe: str = "1h",
    ) -> None:
        self.timeframe = timeframe
        self.market_repo = market_repo or MarketDataRepository()
        self.feature_extractor = feature_extractor or FeatureExtractor(feature_list or [])
        self.agent_params = agent_params or {}
        self.agent: Optional[DQNAgent] = None

    async def collect_data(
        self,
        asset: str,
        platform: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Collect OHLCV data from the repository."""
        return await self.market_repo.get_ohlcv_data(
            exchange=platform,
            symbol=asset,
            timeframe=self.timeframe,
            start_time=start,
            end_time=end,
        )

    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features using the configured feature extractor."""
        return await run_in_threadpool(self.feature_extractor.extract_features, data)

    async def train(
        self,
        asset: str,
        platform: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        episodes: int = 10,
        max_steps: int = 100,
    ) -> None:
        """Train a DQN agent on the specified asset data."""
        market_data = await self.collect_data(asset, platform, start, end)
        features = await self.extract_features(market_data)
        env = MarketEnvironment(market_data, features)

        self.agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **self.agent_params,
        )

        for _ in range(episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update_model()
                state = next_state
                if done:
                    break

    def predict_action(self, state: Any) -> int:
        """Predict the next action for a given state."""
        if not self.agent:
            raise ValueError("Agent not trained")
        return self.agent.select_action(state, test_mode=True)
=======
"""Reinforcement learning training helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from common.logger import get_logger
from data_storage.market_data import MarketDataRepository
from feature_service.feature_extraction import FeatureExtractor
from strategy_brains.reinforcement_brain import TradingEnvironment, DQNAgent

logger = get_logger(__name__)


class PPOAgent:
    """Minimal PPO-style agent for discrete action spaces."""

    def __init__(self, state_shape: tuple, action_dim: int, lr: float = 3e-4):
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(state_shape)), 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self._log_prob = None

    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        if training:
            self._log_prob = dist.log_prob(action)
        return int(action.item())

    def learn(self, reward: float) -> None:
        if self._log_prob is None:
            return
        loss = -(self._log_prob * reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def _create_agent(algo: str, state_shape: tuple, action_dim: int):
    if algo == "ppo":
        return PPOAgent(state_shape, action_dim)
    return DQNAgent(state_shape, action_dim)


async def _train_rl_agent(
    config: Config,
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "1h",
    training_period: str = "1y",
) -> Dict[str, str]:
    """Train a reinforcement learning agent and return metrics."""

    repo = MarketDataRepository(config)
    feature_list = config.get("ml_models.rl_features", [])
    extractor = FeatureExtractor(feature_list)

    raw_df = await repo.get_ohlcv_data(exchange, symbol, timeframe)
    feature_df = extractor.extract_features(raw_df)
    env_df = pd.concat([raw_df, feature_df], axis=1)

    env = TradingEnvironment(env_df)
    state, _ = env.reset()

    algo = config.get("ml_models.rl_algorithm", "dqn").lower()
    agent = _create_agent(algo, state.shape, env.action_space.n)
    episodes = config.get("ml_models.rl_episodes", 10)

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if algo == "ppo":
                agent.learn(reward)
            else:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            state = next_state
        if hasattr(agent, "update_target_model"):
            agent.update_target_model()

    logger.info("Finished RL training", extra={"algorithm": algo, "episodes": episodes})
    return {"status": "success", "algorithm": algo, "episodes": str(episodes)}

