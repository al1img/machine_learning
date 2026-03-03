# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

All scripts use bare module imports (e.g., `from gridworld import GridWorld`), so they must be run from within the `rl/` directory:

```bash
cd rl && python rl.py
```

## Code Style

Configured in `pyproject.toml`:

- Line length: 120 characters (black and pylint)
- Max function args: 10 (pylint)
- Min public methods: 1 (pylint)

## Architecture

The project contains reinforcement learning algorithm implementations applied to a `GridWorld` environment.

### Environment (`rl/gridworld.py`)

`GridWorld` is a 4×4 grid with configurable terminal states, step reward, and terminal reward. Key interface:

- `env.states` — all `(row, col)` tuples
- `env.actions` — list of `Action` enum values (UP/RIGHT/DOWN/LEFT)
- `env.get_transition(state, action)` → `list[Transition]` — supports stochastic transitions via probability-weighted list; currently deterministic (single transition, probability=1.0)
- `env.next_state(state, action)`, `env.reward(state, action, next_state)` — used by model-free agents during episode generation

### Agent Interface (`rl/rl.py`)

All agents follow a consistent interface:

- `agent.train()` → `int` (iteration count)
- `agent.values` property → `dict[State, float]`
- `agent.policy` property → `dict[State, Action]`
- `agent.quality` property (Q-agents only) → `dict[State, dict[Action, float]]`

**Note:** `SARSAAgent`, `QLearningAgent`, `ActorCriticAgent`, and `PolicyGradientAgent` have `train()` returning `tuple[int, dict, dict]` — these are older-style agents that haven't been refactored yet to the property-based interface used in `rl.py`. The unreachable code in `main()` after the `return` shows the intent to integrate them.

### Agent Taxonomy

**Model-based** (require full environment model, use `get_transition()`):

- `viagent.py` — Value Iteration
- `piagent.py` — Policy Iteration

**Model-free, value-based** (learn V(s) or Q(s,a) from episodes):

- `mcvagent.py` — Monte Carlo Value (first-visit, learns V(s), requires model to extract policy)
- `mcqagent.py` — Monte Carlo Q (first-visit, learns Q(s,a), no model needed for policy)
- `tdagent.py` — TD(0) (learns V(s), requires model to extract policy via `calc_best_policy_from_values`)
- `sarsaagent.py` — SARSA (on-policy, learns Q(s,a))
- `qagent.py` — Q-Learning (off-policy, learns Q(s,a))

**Policy-based** (learn policy directly):

- `acagent.py` — Actor-Critic (TD critic + softmax actor with action preferences)
- `pgagent.py` — REINFORCE / Policy Gradient (Monte Carlo returns + softmax policy)

### Shared Utilities (`rl/utils.py`, `rl/common.py`)

`utils.py` provides:

- `format_*` functions — display values, policy (arrows), Q-values (compass layout), state visit counts
- `calc_action_probabilities(actions, best_action, epsilon)` — epsilon-greedy probability distribution
- `calc_best_policy_from_values(env, values, gamma)` — requires model access
- `calc_best_policy_from_quality(q)` — model-free policy extraction
- `calc_values_from_quality(q)` — V(s) = max_a Q(s,a)
- `get_action(probabilities)` — samples action from probability dict using `np.random.choice`

`common.py` defines `EpisodeItem(state, action, reward)` — shared by `mcvagent.py` (others define their own local copy).
