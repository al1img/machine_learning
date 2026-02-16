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

## Links

1. Steve Brunton. ["Reinforcement learning"](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74). Video course.
2. Steve Brunton. ["Reinforcement learning"](https://faculty.washington.edu/sbrunton/databookRL.pdf). Data book.
3. Invisible AI Guru Jii. ["Dynamic Programming, Policy Iteration, and Value Iteration in Reinforcement Learning"](https://medium.com/@apukumargiri1/dynamic-programming-policy-iteration-and-value-iteration-in-reinforcement-learning-675fee67905c). Medium.
4. a7med3laa. ["DRL-Books-resources"](https://github.com/a7med3laa/DRL-Books-resources). Github.
