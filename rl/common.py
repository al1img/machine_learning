""" "Common utilities and classes for reinforcement learning agents."""

from dataclasses import dataclass

from gridworld import Action, State


@dataclass(frozen=True)
class EpisodeItem:
    """Represents an item in an episode."""

    state: State
    action: Action
    reward: float
