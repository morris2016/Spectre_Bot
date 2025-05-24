#!/usr/bin/env python3
"""Simple DQN agent leveraging PyTorch."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Deep Q-Network agent."""

    def __init__(self, state_dim: int, action_dim: int, config: dict | None = None):
        super().__init__(state_dim, action_dim, config)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon = self.config.get("epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 32)
        self.lr = self.config.get("lr", 1e-3)
        self.memory = deque(maxlen=self.config.get("memory_size", 10000))

        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: Any) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    def store_transition(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        q_values = self.model(states_t).gather(1, actions_t).squeeze()
        with torch.no_grad():
            next_q = self.model(next_states_t).max(1)[0]
        target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())
