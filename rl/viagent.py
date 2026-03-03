"""Value Iteration Agent for GridWorld environment."""

from collections import defaultdict

from gridworld import Action, GridWorld, State


class ValueIterationAgent:
    """Value iteration agent to find the optimal policy for a given GridWorld environment."""

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
        self._values: dict[State, float] = defaultdict(float)

    def train(self) -> int:
        """Trains agent."""

        # calculate values

        i = 0

        for i in range(self._max_iters):
            delta = 0.0

            for state in self._env.states:
                if self._env.is_terminal(state):
                    continue

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

                    best_value = max(best_value, value)

                delta = max(delta, abs(self._values[state] - best_value))
                self._values[state] = best_value

            if delta < self._theta:
                break

        return i + 1

    @property
    def values(self) -> dict[State, float]:
        """Returns state values."""
        return dict(self._values)

    @property
    def policy(self) -> dict[State, Action]:
        """Returns the best policy based on the calculated state values."""
        # calculate policy

        policy: dict[State, Action] = {}

        for state in self._env.states:
            if self._env.is_terminal(state):
                continue

            best_action = self._env.actions[0]
            best_value = float("-inf")

            for action in self._env.actions:
                # we can not use direct values[next_state] here because we need to use reward
                # instead of value for policy evaluation for example, terminal state has zero value
                # but valuable reward. See doc/vi_policy_calc.png for more details.

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

            policy[state] = best_action

        return policy
