"""Actor-Critic Agent"""

import random
from collections import defaultdict

import numpy as np
import utils
from gridworld import Action, GridWorld, State


class ActorCriticAgent:
    """Actor-Critic agent to find the optimal policy for a given GridWorld environment."""

    def __init__(
        self,
        env: GridWorld,
        alpha_critic: float = 0.1,
        alpha_actor: float = 0.1,
        gamma: float = 0.99,
        max_steps: int = 100,
        max_iters: int = 1000,
    ) -> None:
        self._env = env
        self._alpha_critic = alpha_critic
        self._alpha_actor = alpha_actor
        self._gamma = gamma
        self._max_steps = max_steps
        self._max_iters = max_iters

        # Critic: state-value function V(s)
        self._values: dict[State, float] = {}
        # Actor: action preferences h(s,a) — softmax over these gives π(a|s)
        self._preferences: dict[State, dict[Action, float]] = defaultdict(lambda: defaultdict(float))

    def train(self) -> int:
        """Trains agent."""

        iters = 0

        for _ in range(self._max_iters):
            state = self._get_start_state()
            I = 1.0

            for i in range(self._max_steps):
                if self._env.is_terminal(state):
                    break

                policy = self._softmax_policy(state)
                action = utils.get_action(policy)

                next_state = self._env.next_state(state, action)
                reward = self._env.reward(state, action, next_state)

                # TD error: δ ← R + γ·V(Sₜ₊₁) - V(Sₜ)  (V(Sₜ₊₁) = 0 if Sₜ₊₁ is terminal)
                v_next = 0.0 if self._env.is_terminal(next_state) else self._values.get(next_state, 0.0)
                td_error = reward + self._gamma * v_next - self._values.get(state, 0.0)

                # Critic update: w ← w + αʷ · δ · ∇v̂(Sₜ, w)
                self._values[state] = self._values.get(state, 0.0) + self._alpha_critic * td_error

                # Actor update: θ ← θ + αᶿ · I · δ · ∇ln π(Aₜ|Sₜ, θ)
                # Full softmax gradient: ∂ln π(Aₜ|s)/∂h(s,a) = 1{a=Aₜ} - π(a|s)
                for a in self._env.actions:
                    grad = (1.0 if a == action else 0.0) - policy[a]
                    self._preferences[state][a] += self._alpha_actor * I * td_error * grad

                I *= self._gamma
                state = next_state

            iters += i

        return iters

    @property
    def values(self) -> dict[State, float]:
        """Returns state value estimates."""
        return self._values

    @property
    def policy(self) -> dict[State, Action]:
        """Returns the greedy policy derived from action preferences."""
        return utils.calc_best_policy_from_quality(self._preferences)

    @property
    def quality(self) -> dict[State, dict[Action, float]]:
        """Returns action preferences h(s,a) for each state-action pair."""
        return self._preferences

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

    def _get_start_state(self) -> State:
        """Gets a random non-terminal state to start an episode."""

        return (random.randint(0, self._env.size[0] - 1), random.randint(0, self._env.size[1] - 1))
