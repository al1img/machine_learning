"""SARSA Agent"""

import random
from collections import defaultdict

import utils
from gridworld import Action, GridWorld, State


class SARSAAgent:
    """SARSA agent to find the optimal policy for a given GridWorld environment."""

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

        # we can store only best action for each state and implement e-greedy policy in _generate_episode as:
        # random(epsilon) -> random action, else -> best action. But for generality, we will store the probabilities of
        # taking each action in each state, which allows us to implement more complex policies if needed.
        self._policy: dict[State, dict[Action, float]] = {}

    def train(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        """Trains agent."""

        iters = 0
        q: dict[State, dict[Action, float]] = defaultdict(lambda: defaultdict(float))

        for _ in range(self._max_iters):
            state = self._get_start_state()
            action = utils.get_action(self._policy.get(state, utils.calc_action_probabilities(self._env.actions)))

            for i in range(self._max_steps):
                if self._env.is_terminal(state):
                    break

                next_state = self._env.next_state(state, action)
                next_action = utils.get_action(
                    self._policy.get(next_state, utils.calc_action_probabilities(self._env.actions))
                )
                reward = self._env.reward(state, action, next_state)

                q[state][action] += self._alpha * (reward + self._gamma * q[next_state][next_action] - q[state][action])

                state = next_state
                action = next_action

            best_policy = utils.calc_best_policy_from_quality(q)

            for state, action in best_policy.items():
                self._policy[state] = utils.calc_action_probabilities(self._env.actions, action, self._epsilon)

            iters += i

        return iters, utils.calc_value_from_quality(q), best_policy

    def _get_start_state(self) -> State:
        """Gets a random non-terminal state to start an episode."""

        # We start from a random non-terminal state to ensure that we explore the state space effectively.
        # If it is allowed by the environment, we could also start from a fixed state or a distribution of states.

        return (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
