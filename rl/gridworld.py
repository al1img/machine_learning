"""A simple grid world environment for reinforcement learning."""

from dataclasses import dataclass
from enum import IntEnum

State = tuple[int, int]


class Action(IntEnum):
    """Actions for the grid world environment."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen=True)
class Transition:
    """Represents a transition in the grid world environment."""

    probability: float
    next_state: State
    reward: float


class GridWorld:
    """A simple grid world environment for reinforcement learning."""

    def __init__(
        self,
        size: tuple[int, int] = (4, 4),
        terminal_states: tuple[State, ...] = ((3, 3),),
        step_reward: float = 0.0,
        terminal_reward: float = 1.0,
    ) -> None:
        self._size = size
        self._terminal_states = set(terminal_states)
        self._step_reward = step_reward
        self._terminal_reward = terminal_reward

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of the grid world."""
        return self._size

    @property
    def states(self) -> list[State]:
        """Returns a list of all states in the grid world."""
        return [(r, c) for r in range(self._size[0]) for c in range(self._size[1])]

    @property
    def actions(self) -> list[Action]:
        """Returns a list of all possible actions."""
        return list(Action)

    def is_terminal(self, state: State) -> bool:
        """Returns True if the given state is a terminal state."""
        return state in self._terminal_states

    def next_state(self, state: State, action: Action) -> State:
        """Returns the next state given the current state and action."""
        if self.is_terminal(state):
            return state

        r, c = state

        if action == Action.UP:
            r = max(r - 1, 0)
        elif action == Action.RIGHT:
            c = min(c + 1, self._size[1] - 1)
        elif action == Action.DOWN:
            r = min(r + 1, self._size[0] - 1)
        elif action == Action.LEFT:
            c = max(c - 1, 0)

        return (r, c)

    def reward(self, cur_state: State, action: Action, next_state: State) -> float:
        """
        Returns the reward for taking the given action in the current state and transitioning to the next state.
        This is general case where the reward can depend on the current state, action, and next state.
        In this simple grid world, we will return a fixed step reward for non-terminal transitions and the terminal
        reward for transitions into terminal states.
        """
        _ = action  # Unused in this simple implementation, but included for generality.

        if self.is_terminal(cur_state):
            return 0.0

        if self.is_terminal(next_state):
            return self._terminal_reward

        return self._step_reward

    def get_transition(self, state: State, action: Action) -> list[Transition]:
        """Returns a list of possible transitions for the given state and action."""

        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)

        # In this simple grid world, we assume deterministic transitions, so we return a single
        # transition with probability 1.0. In a more complex environment, this method could return
        # multiple transitions with different probabilities.
        return [Transition(1.0, next_state, reward)]
