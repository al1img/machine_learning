"""Utils for the GridWorld environment agents."""

import numpy as np
from common import EpisodeItem
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


def format_state_counts(state_counts: dict[State, int], env: GridWorld) -> str:
    """Formats state counts as a grid of numbers."""
    rows = []

    for r in range(env.size[0]):
        row = []

        for c in range(env.size[1]):
            state = (r, c)
            row.append(f"{state_counts.get(state, 0):6d}")

        rows.append(" ".join(row))

    return "\n".join(rows)


def format_quality(q: dict[State, dict[Action, float]], env: GridWorld) -> str:
    """Formats quality (Q) function as a grid showing action values per state.

    Each cell shows Q(s, a) for all 4 actions in a compass layout:
        ↑
      ←   →
        ↓
    """
    val_w = 6
    cell_w = val_w * 2 + 1  # left(6) + sep(1) + right(6) = 13

    rows = []

    for r in range(env.size[0]):
        top_row, mid_row, bot_row = [], [], []

        for c in range(env.size[1]):
            state = (r, c)
            actions = q.get(state, {})

            if env.is_terminal(state):
                top_row.append(" " * cell_w)
                mid_row.append(f"{'T':^{cell_w}}")
                bot_row.append(" " * cell_w)
            else:
                up = f"{actions.get(Action.UP,    0.0):{val_w}.3f}"
                down = f"{actions.get(Action.DOWN,  0.0):{val_w}.3f}"
                left = f"{actions.get(Action.LEFT,  0.0):{val_w}.3f}"
                right = f"{actions.get(Action.RIGHT, 0.0):{val_w}.3f}"

                top_row.append(f"{up:^{cell_w}}")
                mid_row.append(f"{left} {right}")
                bot_row.append(f"{down:^{cell_w}}")

        rows.append(" ".join(top_row))
        rows.append(" ".join(mid_row))
        rows.append(" ".join(bot_row))

        if r < env.size[0] - 1:
            rows.append("")

    return "\n".join(rows)


def calc_action_probabilities(
    actions: list[Action], best_action: Action = None, epsilon: float = 1.0
) -> dict[Action, float]:
    """Calculates the probabilities of taking each action based on the epsilon-greedy policy."""
    probabilities = {}

    p = (epsilon / len(actions)) if best_action is not None else (1.0 / len(actions))

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


def calc_values_from_quality(q: dict[State, dict[Action, float]]) -> dict[State, float]:
    """Calculates state values from action values."""

    values = {}

    for state, actions in q.items():
        values[state] = max(actions.values(), default=0.0)

    return values


def calc_best_policy_from_quality(q: dict[State, dict[Action, float]]) -> dict[State, Action]:
    """Calculates the best policy based on action values."""

    best_policy = {}

    for state, actions in q.items():
        best_action = max(actions, key=actions.get, default=None)
        best_policy[state] = best_action

    return best_policy


def get_action(probabilities: dict[Action, float]) -> Action:
    """Selects an action based on the given probabilities."""
    actions = list(probabilities.keys())
    probs = list(probabilities.values())

    return Action(np.random.choice(actions, p=probs))


def calc_returns(episode: list[EpisodeItem], gamma: float) -> list[float]:
    """Computes discounted returns Gₜ = Σₖ₌ₜ₊₁ᵀ γᵏ⁻ᵗ⁻¹ · Rₖ for each step."""
    returns = [0.0] * len(episode)
    g = 0.0

    for t in reversed(range(len(episode))):
        g = episode[t].reward + gamma * g
        returns[t] = g

    return returns
