"""Monte Carlo Agent for Reinforcement Learning"""

import random
from collections import defaultdict
from dataclasses import dataclass

import utils
from gridworld import Action, GridWorld, State


@dataclass(frozen=True)
class EpisodeItem:
    """Represents an item in an episode."""

    state: State
    action: Action
    reward: float


class MonteCarloQAgent:
    """Monte Carlo Q agent to find the optimal policy for a given GridWorld environment."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.25,
        max_steps: int = 100,
        max_iters: int = 1000,
    ) -> None:
        self._env = env
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._max_steps = max_steps
        self._max_iters = max_iters
        self._q: dict[State, dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        self._state_counts: dict[State, int] = defaultdict(int)

        # we can store only best action for each state and implement e-greedy policy in _generate_episode as:
        # random(epsilon) -> random action, else -> best action. But for generality, we will store the probabilities of
        # taking each action in each state, which allows us to implement more complex policies if needed.
        self._policy: dict[State, dict[Action, float]] = {}

    def train(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        """Trains agent."""
        iters = 0

        for _ in range(self._max_iters):
            steps, episode = self._generate_episode()
            iters += steps

            g = 0.0

            for t in reversed(range(len(episode))):
                cur_item = episode[t]
                g = self._gamma * g + cur_item.reward

                if not any((item.state == cur_item.state and item.action == cur_item.action) for item in episode[:t]):
                    self._q[cur_item.state][cur_item.action] += self._alpha * (
                        g - self._q[cur_item.state][cur_item.action]
                    )

                    q_actions = self._q[cur_item.state]
                    best_action = max(q_actions, key=q_actions.get)

                    self._policy[cur_item.state] = utils.calc_action_probabilities(
                        self._env.actions, best_action, self._epsilon
                    )

        return iters

    @property
    def values(self) -> dict[State, float]:
        """Returns the values of states."""

        return utils.calc_values_from_quality(self._q)

    @property
    def policy(self) -> dict[State, Action]:
        """Returns the best action for each state according to the current policy."""

        return utils.calc_best_policy_from_quality(self._q)

    @property
    def state_counts(self) -> dict[State, int]:
        """Returns the number of times each state was visited during training."""
        return dict(self._state_counts)

    @property
    def quality(self) -> dict[State, dict[Action, float]]:
        """Returns the quality of each state-action pair."""
        return dict(self._q)

    def _generate_episode(self) -> tuple[int, list[EpisodeItem]]:
        """Generates an episode by following the current policy."""

        episode: list[EpisodeItem] = []

        state = self._get_start_state()

        i = 0

        for i in range(self._max_steps):
            if self._env.is_terminal(state):
                break

            self._state_counts[state] += 1

            action = utils.get_action(self._policy.get(state, utils.calc_action_probabilities(self._env.actions)))
            next_state = self._env.next_state(state, action)
            reward = self._env.reward(state, action, next_state)
            episode.append(EpisodeItem(state, action, reward))
            state = next_state

        return i + 1, episode

    def _get_start_state(self) -> State:
        """Gets a random non-terminal state to start an episode."""

        # We start from a random non-terminal state to ensure that we explore the state space effectively.
        # If it is allowed by the environment, we could also start from a fixed state or a distribution of states.

        return (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
