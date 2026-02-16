"""Policy Iteration Agent for GridWorld environment."""

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
        self._policy: dict[State, Action] = {}

    def train(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        """Trains agent."""
        values = {state: 0.0 for state in self._env.states}
        iters = 0

        for state in self._env.states:
            if not self._env.is_terminal(state):
                self._policy[state] = self._env.actions[0]

        while True:
            i, values = self._evaluate_policy(values)

            iters += i

            if self._improve_policy(values):
                break

        return iters, values, self._policy

    def _evaluate_policy(self, values: dict[State, float]) -> tuple[int, dict[State, float]]:
        for i in range(self._max_iters):
            delta = 0.0

            for state in self._env.states:
                if self._env.is_terminal(state):
                    continue

                action = self._policy[state]
                next_state = self._env.next_state(state, action)
                reward = self._env.reward(next_state)
                value = reward + self._gamma * values[next_state]

                delta = max(delta, abs(values[state] - value))
                values[state] = value

            if delta < self._theta:
                break

        return i + 1, values

    def _improve_policy(self, values: dict[State, float]) -> bool:
        policy_stable = True

        for state in self._env.states:
            if self._env.is_terminal(state):
                continue

            old_action = self._policy[state]
            best_action = old_action
            best_value = float("-inf")

            for action in self._env.actions:
                next_state = self._env.next_state(state, action)
                reward = self._env.reward(next_state)
                value = reward + self._gamma * values[next_state]

                if value > best_value:
                    best_value = value
                    best_action = action

            self._policy[state] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy_stable
