"""Value Iteration Agent for GridWorld environment."""

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

    def train(self) -> tuple[int, dict[State, float], dict[State, Action]]:
        """Trains agent."""

        # calculate values

        values = {state: 0.0 for state in self._env.states}

        for i in range(self._max_iters):
            delta = 0.0

            for state in self._env.states:
                if self._env.is_terminal(state):
                    continue

                best_value = float("-inf")

                for action in self._env.actions:
                    next_state = self._env.next_state(state, action)
                    reward = self._env.reward(next_state)

                    value = reward + self._gamma * values[next_state]
                    best_value = max(best_value, value)

                delta = max(delta, abs(values[state] - best_value))
                values[state] = best_value

            if delta < self._theta:
                break

        # calculate policy

        policy: dict[State, Action] = {}

        for state in self._env.states:
            if self._env.is_terminal(state):
                continue

            best_action = self._env.actions[0]
            best_value = float("-inf")

            for action in self._env.actions:
                next_state = self._env.next_state(state, action)
                reward = self._env.reward(next_state)

                # we can not use direct values[next_state] here because we need to use reward instead of value for policy evaluation
                # for example, terminal state has zero value but valuable reward. See doc/vi_policy_calc.png for more details.

                value = reward + self._gamma * values[next_state]

                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action

        return i + 1, values, policy
