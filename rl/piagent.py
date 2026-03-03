"""Policy Iteration Agent for GridWorld environment."""

from collections import defaultdict

from gridworld import Action, GridWorld, State


class PolicyIterationAgent:
    """Policy iteration agent to find the optimal policy for a given GridWorld environment."""

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iters: int = 1000,
    ) -> None:
        self._env = env
        self._gamma = gamma
        self._theta = theta
        self._max_iters = max_iters
        self._policy: dict[State, Action] = defaultdict(lambda: self._env.actions[0])
        self._values: dict[State, float] = defaultdict(float)

    def train(self) -> int:
        """Trains agent."""
        total_iters = 0

        for state in self._env.states:
            if not self._env.is_terminal(state):
                self._policy[state] = self._env.actions[0]

        while True:
            num_iters = self._evaluate_policy()

            total_iters += num_iters

            if self._improve_policy():
                break

        return total_iters

    @property
    def values(self) -> dict[State, float]:
        """Returns state values."""
        return dict(self._values)

    @property
    def policy(self) -> dict[State, Action]:
        """Returns policy."""
        return dict(self._policy)

    def _evaluate_policy(self) -> int:
        i = 0

        for i in range(self._max_iters):
            delta = 0.0

            for state in self._env.states:
                if self._env.is_terminal(state):
                    continue

                action = self._policy[state]

                # For simple deterministic environment we can use direct next state and reward
                # calculation here:
                # next_state = self._env.next_state(state, action)
                # reward = self._env.reward(state, action, next_state)
                # value = reward + self._gamma * values[next_state]

                # But for more complex environment we need to calculate value using transition
                # probabilities:

                value = 0.0

                for transition in self._env.get_transition(state, action):
                    value += transition.probability * (
                        transition.reward + self._gamma * self._values[transition.next_state]
                    )

                delta = max(delta, abs(self._values[state] - value))
                self._values[state] = value
            if delta < self._theta:
                break

        return i + 1

    def _improve_policy(self) -> bool:
        policy_stable = True

        for state in self._env.states:
            if self._env.is_terminal(state):
                continue

            old_action = self._policy[state]
            best_action = old_action
            best_value = float("-inf")

            for action in self._env.actions:
                # For simple deterministic environment we can use direct next state and reward
                # calculation here:
                # next_state = self._env.next_state(state, action)
                # reward = self._env.reward(state, action, next_state)
                # value = reward + self._gamma * values[next_state]

                # But for more complex environment we need to calculate value using transition
                # probabilities:

                value = 0.0

                for transition in self._env.get_transition(state, action):
                    value += transition.probability * (
                        transition.reward + self._gamma * self._values[transition.next_state]
                    )

                if value > best_value:
                    best_value = value
                    best_action = action

            self._policy[state] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy_stable
