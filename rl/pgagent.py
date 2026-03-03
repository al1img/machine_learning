"""Policy Gradient (REINFORCE) Agent"""

import random
from collections import defaultdict

import numpy as np
import utils
from common import EpisodeItem
from gridworld import Action, GridWorld, State


class PolicyGradientAgent:
    """REINFORCE (Monte Carlo Policy Gradient) agent to find the optimal policy for a given GridWorld environment."""

    def __init__(
        self,
        env: GridWorld,
        alpha: float = 0.1,
        gamma: float = 0.99,
        max_steps: int = 100,
        max_iters: int = 1000,
    ) -> None:
        self._env = env
        self._alpha = alpha
        self._gamma = gamma
        self._max_steps = max_steps
        self._max_iters = max_iters

        self._values: dict[State, float] = {}  # state value estimates for reporting
        # Action preferences h(s,a) — softmax over these gives π(a|s)
        self._preferences: dict[State, dict[Action, float]] = defaultdict(lambda: defaultdict(float))

    def train(self) -> int:
        """Trains agent using REINFORCE algorithm."""

        iters = 0

        for _ in range(self._max_iters):
            steps, episode = self._generate_episode()
            iters += steps

            # Compute returns for each step
            returns = self._compute_returns(episode)

            for t, item in enumerate(episode):
                policy = self._softmax_policy(item.state)
                g = returns[t]

                # Update state value estimate (for reporting)
                self._values[item.state] = self._values.get(item.state, 0.0) + self._alpha * (
                    g - self._values.get(item.state, 0.0)
                )

                # Policy gradient update: h(s,a) ← h(s,a) + α · γᵗ · Gₜ · (1 - π(a|s))
                # The (1 - π(a|s)) term is the softmax policy gradient ∇ln π(a|s) for the selected action
                self._preferences[item.state][item.action] += (
                    self._alpha * (self._gamma**t) * g * (1 - policy[item.action])
                )

        return iters

    @property
    def values(self) -> dict[State, float]:
        """Returns state value estimates."""
        return self._values

    @property
    def policy(self) -> dict[State, dict[Action, float]]:
        """Returns the current policy π(a|s) for each state."""
        return utils.calc_best_policy_from_quality(self._preferences)

    @property
    def quality(self) -> dict[State, dict[Action, float]]:
        """Returns action preferences h(s,a) for each state-action pair."""
        return self._preferences

    def _compute_returns(self, episode: list[EpisodeItem]) -> list[float]:
        """Computes discounted returns Gₜ for each step in the episode."""

        returns = [0.0] * len(episode)
        g = 0.0

        for t in reversed(range(len(episode))):
            g = episode[t].reward + self._gamma * g
            returns[t] = g

        return returns

    def _softmax_policy(self, state: State) -> dict[Action, float]:
        """Computes π(·|s) via softmax over action preferences h(s, ·)."""

        prefs = self._preferences[state]
        actions = self._env.actions
        h = np.array([prefs[a] for a in actions])

        # Numerically stable softmax
        h = h - np.max(h)
        exp_h = np.exp(h)
        probs = exp_h / np.sum(exp_h)

        return {a: float(p) for a, p in zip(actions, probs)}

    def _generate_episode(self) -> tuple[int, list[EpisodeItem]]:
        """Generates an episode by following the current policy."""

        episode: list[EpisodeItem] = []
        state = self._get_start_state()
        i = 0

        for i in range(self._max_steps):
            if self._env.is_terminal(state):
                break

            policy = self._softmax_policy(state)
            action = utils.get_action(policy)

            next_state = self._env.next_state(state, action)
            reward = self._env.reward(state, action, next_state)
            episode.append(EpisodeItem(state, action, reward))
            state = next_state

        return i + 1, episode

    def _get_start_state(self) -> State:
        """Gets a random non-terminal state to start an episode."""

        return (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
