"""Utils for the GridWorld environment agents."""

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
                row.append(arrows[policy[state]])

        rows.append(" ".join(row))

    return "\n".join(rows)


def format_values(values: dict[State, float], env: GridWorld) -> str:
    """Formats values as a grid of numbers."""
    rows = []

    for r in range(env.size[0]):
        row = []

        for c in range(env.size[1]):
            state = (r, c)
            row.append(f"{values[state]:6.2f}")

        rows.append(" ".join(row))

    return "\n".join(rows)
