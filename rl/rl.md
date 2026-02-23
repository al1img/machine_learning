# Reinforcement learning

## Model-based

### Value iteration

Best when:

* state space is larger — you want to avoid the cost of full policy evaluation;
* early policy evaluation sweeps provide diminishing returns (values change little after the first few sweeps);
* you want simpler implementation.

Trade-off:

* more outer iterations, but each iteration is cheaper.

### Policy iteration

Best when:

* state space is relatively small — each evaluation sweep is cheap;
* policy converges in few improvement steps (typically very few iterations, often < 10);
* you need an explicit policy at every stage.

Trade-off:

* each iteration is expensive (full evaluation until convergence), but fewer outer iterations are needed.

## Model-free

### Monte Carlo

* If model is not available better to estimate quality function (Q) as it is hard to calculate policy.
* Instead of averaging value, learning rate `alpha` could be used.

### TD, SARSA and Q-Learning

* DP, TD, and Monte Carlo methods all use some variation of generalized policy iteration (GPI).

## Comparison

| Algorithm      | What it learns | Update rule                                                                   | Policy         |
|----------------|----------------|-------------------------------------------------------------------------------|----------------|
| **TD(0)**      | $V(s)$         | $V(s) \leftarrow V(s) + \alpha [R + \gamma V(s') - V(s)]$                     | Requires model |
| **SARSA**      | $Q(s, a)$      | $Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(s', a') - Q(s,a)]$           | On-policy      |
| **Q-Learning** | $Q(s, a)$      | $Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s,a)]$ | Off-policy     |

## Key Differences

* **SARSA** name comes from the tuple $(S, A, R, S', A')$ — the next action $a'$ is chosen by the **current policy** (on-policy).
* **Q-Learning** uses $\max_{a'} Q(s', a')$ regardless of the action actually taken (off-policy).
* **TD(0)** learns state values $V(s)$ and requires a model (e.g., `env.next_state()`, `env.reward()`) to derive the policy.

## Links

1. Steve Brunton. ["Reinforcement learning"](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74). Video course.
2. Steve Brunton. ["Reinforcement learning"](https://faculty.washington.edu/sbrunton/databookRL.pdf). Data book.
3. Richard S. Sutton and Andrew G. Barto ["Reinforcement Learning: An Introduction"](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
4. Invisible AI Guru Jii. ["Dynamic Programming, Policy Iteration, and Value Iteration in Reinforcement Learning"](https://medium.com/@apukumargiri1/dynamic-programming-policy-iteration-and-value-iteration-in-reinforcement-learning-675fee67905c). Medium.
5. a7med3laa. ["DRL-Books-resources"](https://github.com/a7med3laa/DRL-Books-resources). Github.
6. Invisible AI Guru Jii. ["Monte Carlo Methods in Reinforcement Learning"](https://medium.com/@apukumargiri1/monte-carlo-methods-in-reinforcement-learning-04a8e406b848)
