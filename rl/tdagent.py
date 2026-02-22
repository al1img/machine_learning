"""Temporal Difference Agent"""

import random

import utils
from gridworld import Action, GridWorld, State


class TemporalDifferenceAgent:
    """Temporal Difference agent to find the optimal policy for a given GridWorld environment."""

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
        values: dict[State, float] = {}

        for _ in range(self._max_iters):
            state = self._get_start_state()

            for i in range(self._max_steps):
                if self._env.is_terminal(state):
                    break

                action = utils.get_action(self._policy.get(state, utils.calc_action_probabilities(self._env.actions)))
                next_state = self._env.next_state(state, action)
                reward = self._env.reward(state, action, next_state)

                values[state] = values.get(state, 0.0) + self._alpha * (
                    reward + self._gamma * values.get(next_state, 0.0) - values.get(state, 0.0)
                )

                state = next_state

            # If model is available, we can derive the policy from the values. In this simple grid world,
            # we can just check all possible actions and choose the one that leads to the state with the highest value.
            best_policy = utils.calc_best_policy_from_values(self._env, values, self._gamma)

            for state, action in best_policy.items():
                self._policy[state] = utils.calc_action_probabilities(self._env.actions, action, self._epsilon)

            iters += i

        return iters, values, best_policy

    def _get_start_state(self) -> State:
        """Gets a random non-terminal state to start an episode."""

        # We start from a random non-terminal state to ensure that we explore the state space effectively.
        # If it is allowed by the environment, we could also start from a fixed state or a distribution of states.

        return (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
