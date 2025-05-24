#!/usr/bin/env python3
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

