"""Utils for the GridWorld environment agents."""

import numpy as np
from gridworld import Action, GridWorld, State


def format_policy(policy: dict[State, Action], env: GridWorld) -> str:
    """Formats policy as a grid of arrows."""
    arrows = {Action.UP: "↑", Action.RIGHT: "→", Action.DOWN: "↓", Action.LEFT: "←"}

    rows = []

    for r in range(env.size[0]):
        row = []

        for c in range(env.size[1]):
            state = (r, c)

            if env.is_terminal(state):
                row.append("T")
            else:
                row.append(arrows[policy.get(state, Action.UP)])

        rows.append(" ".join(row))

    return "\n".join(rows)


def format_values(values: dict[State, float], env: GridWorld) -> str:
    """Formats values as a grid of numbers."""
    rows = []

    for r in range(env.size[0]):
        row = []

        for c in range(env.size[1]):
            state = (r, c)
            row.append(f"{values.get(state, 0.0):6.3f}")

        rows.append(" ".join(row))

    return "\n".join(rows)


def calc_action_probabilities(
    actions: list[Action], best_action: Action = None, epsilon: float = 1.0
) -> dict[Action, float]:
    """Calculates the probabilities of taking each action based on the epsilon-greedy policy."""
    probabilities = {}

    p = epsilon / len(actions)

    for action in actions:
        probabilities[action] = 1 - epsilon + p if best_action is not None and action == best_action else p

    return probabilities


def calc_best_policy_from_values(env: GridWorld, values: dict[State, float], gamma: float) -> dict[State, Action]:
    """Calculates the best policy based on the given state values."""

    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            continue

        best_action = None
        best_value = float("-inf")

        for action in env.actions:
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            value = reward + gamma * values.get(next_state, 0.0)

            if value > best_value:
                best_value = value
                best_action = action

        policy[state] = best_action

    return policy


def calc_value_from_quality(q: dict[State, dict[Action, float]]) -> dict[State, float]:
    """Calculates state values from action values."""

    values = {}

    for state, actions in q.items():
        values[state] = max(actions.values())

    return values


def calc_best_policy_from_quality(q: dict[State, dict[Action, float]]) -> dict[State, Action]:
    """Calculates the best policy based on action values."""

    best_policy = {}

    for state, actions in q.items():
        best_action = max(actions, key=actions.get)
        best_policy[state] = best_action

    return best_policy


def get_action(probabilities: dict[Action, float]) -> Action:
    """Selects an action based on the given probabilities."""
    actions = list(probabilities.keys())
    probs = list(probabilities.values())

    return Action(np.random.choice(actions, p=probs))
