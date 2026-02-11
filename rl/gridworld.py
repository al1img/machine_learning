"""A simple grid world environment for reinforcement learning."""

from enum import IntEnum

State = tuple[int, int]


class Action(IntEnum):
    """Actions for the grid world environment."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


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

    def reward(self, state: State) -> float:
        """Returns state reward."""
        if self.is_terminal(state):
            return self._terminal_reward

        return self._step_reward
