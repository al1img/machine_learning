"""Reinforcement learning algorithms for GridWorld environment."""

from gridworld import GridWorld
from piagent import PolicyIterationAgent
from utils import format_policy, format_values
from viagent import ValueIterationAgent


def main() -> None:
    env = GridWorld()
    agent = ValueIterationAgent(env)
    iters, values, policy = agent.train()

    print(f"Value Iteration converged in {iters} iterations.\n")

    print("\nValue Iteration Values:\n")
    print(format_values(values, env))

    print("\nValue Iteration Policy:\n")
    print(format_policy(policy, env))

    agent = PolicyIterationAgent(env)
    iters, values, policy = agent.train()

    print(f"\nPolicy Iteration converged in {iters} iterations.\n")

    print("\nPolicy Iteration Values:\n")
    print(format_values(values, env))

    print("\nPolicy Iteration Policy:\n")
    print(format_policy(policy, env))


if __name__ == "__main__":
    main()
