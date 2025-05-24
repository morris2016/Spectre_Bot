#!/usr/bin/env python3
"""Reinforcement learning base agent."""

from __future__ import annotations

import abc
from typing import Any
import torch


class BaseAgent(abc.ABC):
    """Abstract base class for RL agents."""

    def __init__(self, state_dim: int, action_dim: int, config: dict | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}

    @abc.abstractmethod
    def select_action(self, state: Any) -> int:
        """Select an action given a state."""

    @abc.abstractmethod
    def store_transition(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Store experience in replay buffer."""

    @abc.abstractmethod
    def update_model(self) -> float | None:
        """Update internal model and return loss."""

    def save_model(self, path: str) -> None:
        torch.save(self, path)

    def load_model(self, path: str) -> None:
        loaded = torch.load(path)
        self.__dict__.update(loaded.__dict__)
