"""Monte Carlo Agent for Reinforcement Learning"""

import random
from dataclasses import dataclass

import numpy as np
from gridworld import Action, GridWorld, State


@dataclass(frozen=True)
class EpisodeItem:
    """Represents an item in an episode."""

    state: State
    action: Action
    reward: float


class MonteCarloAgent:
    """Monte Carlo agent to find the optimal policy for a given GridWorld environment."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.25,
        max_steps: int = 100,
        max_iters: int = 1000,
        q_learning: bool = True,
    ) -> None:
        self._env = env
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._max_steps = max_steps
        self._max_iters = max_iters
        self._q_learning = q_learning

        # we can store only best action for each state and implement e-greedy policy in _generate_episode as:
        # random(epsilon) -> random action, else -> best action. But for generality, we will store the probabilities of
        # taking each action in each state, which allows us to implement more complex policies if needed.
        self._policy: dict[State, dict[Action, float]] = {}

    def train(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        """Trains agent."""

        return self._train_quality_based() if self._q_learning else self._train_value_based()

    def _train_value_based(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        iters = 0
        values_sum: dict[State, float] = {}
        values: dict[State, float] = {}
        counts: dict[State, int] = {}

        for _ in range(self._max_iters):
            steps, episode = self._generate_episode()
            iters += steps

            g = 0.0

            for t in reversed(range(len(episode))):
                cur_item = episode[t]
                g = self._gamma * g + cur_item.reward

                # We implement the first-visit Monte Carlo method, so we only update the value
                # if it is the first time we have visited it in this episode.
                if not any((item.state == cur_item.state) for item in episode[:t]):
                    values_sum[cur_item.state] = values_sum.get(cur_item.state, 0.0) + g
                    counts[cur_item.state] = counts.get(cur_item.state, 0) + 1

            for state, val_sum in values_sum.items():
                values[state] = val_sum / counts[state]

            # If model is available, we can derive the policy from the values. In this simple grid world,
            # we can just check all possible actions and choose the one that leads to the state with the highest value.
            best_policy = self._calc_best_policy_from_values(values)

            for state, action in best_policy.items():
                self._policy[state] = self._get_action_probabilities(action)

        return iters, values, best_policy

    def _calc_best_policy_from_values(self, values: dict[State, float]) -> dict[State, Action]:
        policy = {}

        for state in self._env.states:
            if self._env.is_terminal(state):
                continue

            best_action = None
            best_value = float("-inf")

            for action in self._env.actions:
                next_state = self._env.next_state(state, action)
                reward = self._env.reward(state, action, next_state)
                value = reward + self._gamma * values.get(next_state, 0.0)

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action

        return policy

    def _train_quality_based(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        iters = 0
        q: dict[State, dict[Action, float]] = {}

        for _ in range(self._max_iters):
            steps, episode = self._generate_episode()
            iters += steps

            g = 0.0

            for t in reversed(range(len(episode))):
                cur_item = episode[t]
                g = self._gamma * g + cur_item.reward

                if not any((item.state == cur_item.state and item.action == cur_item.action) for item in episode[:t]):
                    if cur_item.state not in q:
                        q[cur_item.state] = {}

                    q_value = q[cur_item.state].get(cur_item.action, 0.0)
                    q[cur_item.state][cur_item.action] = q_value + self._alpha * (g - q_value)

                    q_actions = q[cur_item.state]
                    best_action = max(q_actions, key=q_actions.get)

                    self._policy[cur_item.state] = self._get_action_probabilities(best_action)

        return iters, self._calc_value_from_quality(q), self._get_best_policy_from_quality(q)

    def _calc_value_from_quality(self, q: dict[State, dict[Action, float]]) -> dict[State, float]:
        values = {}

        for state, actions in q.items():
            values[state] = max(actions.values())

        return values

    def _get_best_policy_from_quality(self, q: dict[State, dict[Action, float]]) -> dict[State, Action]:
        best_policy = {}

        for state, actions in q.items():
            best_action = max(actions, key=actions.get)
            best_policy[state] = best_action

        return best_policy

    def _generate_episode(self) -> tuple[int, list[EpisodeItem]]:
        """Generates an episode by following the current policy."""
        episode: list[EpisodeItem] = []

        # We start from a random non-terminal state to ensure that we explore the state space effectively.
        # If it is allowed by the environment, we could also start from a fixed state or a distribution of states.
        while True:
            state = (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
            if not self._env.is_terminal(state):
                break

        for i in range(self._max_steps):
            action = (
                Action(np.random.choice(list(self._policy[state].keys()), p=list(self._policy[state].values())))
                if state in self._policy
                else random.choice(self._env.actions)
            )

            next_state = self._env.next_state(state, action)
            reward = self._env.reward(state, action, next_state)
            episode.append(EpisodeItem(state, action, reward))
            state = next_state

            if self._env.is_terminal(state):
                break

        return i + 1, episode

    def _get_action_probabilities(self, best_action: Action) -> dict[Action, float]:
        """Calculates the probabilities of taking each action based on the epsilon-greedy policy."""
        probabilities = {}

        for action in self._env.actions:
            probabilities[action] = (
                1 - self._epsilon + self._epsilon / len(self._env.actions)
                if action == best_action
                else self._epsilon / len(self._env.actions)
            )

        return probabilities
