from abc import ABC, abstractmethod
from typing import Any


class RLAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    @abstractmethod
    def select_action(self, state: Any, test_mode: bool = False) -> Any:
        """Return action given state."""

    @abstractmethod
    def train_step(self) -> Any:
        """Perform one training step."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model parameters to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model parameters from path."""
