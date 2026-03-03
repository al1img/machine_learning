"""Reinforcement learning algorithms for GridWorld environment."""

from acagent import ActorCriticAgent
from gridworld import GridWorld
from mcqagent import MonteCarloQAgent
from mcvagent import MonteCarloValueAgent
from pgagent import PolicyGradientAgent
from piagent import PolicyIterationAgent
from qagent import QLearningAgent
from sarsaagent import SARSAAgent
from tdagent import TemporalDifferenceAgent
from utils import format_policy, format_quality, format_values
from viagent import ValueIterationAgent


def main() -> None:
    env = GridWorld()

    agents = (
        ValueIterationAgent(env),
        PolicyIterationAgent(env),
        MonteCarloValueAgent(env),
        MonteCarloQAgent(env),
        TemporalDifferenceAgent(env),
        SARSAAgent(env),
        QLearningAgent(env),
        PolicyGradientAgent(env),
    )

    for agent in agents:
        iters = agent.train()

        print("\n==============================================================================")
        print(f"{agent.__class__.__name__} converged in {iters} iterations.")
        print("==============================================================================")

        if hasattr(agent, "quality"):
            print("\nQuality:\n")
            print(format_quality(agent.quality, env))

        print("\nValues:\n")
        print(format_values(agent.values, env))

        print("\nPolicy:\n")
        print(format_policy(agent.policy, env))

    return

    agent = QLearningAgent(env)
    iters, values, policy = agent.train()

    print(f"\nQ-Learning converged in {iters} iterations.\n")

    print("\nQ-Learning Values:\n")
    print(format_values(values, env))

    print("\nQ-Learning Policy:\n")
    print(format_policy(policy, env))

    agent = ActorCriticAgent(env)
    iters, values, policy = agent.train()

    print(f"\nActor-Critic converged in {iters} iterations.\n")

    print("\nActor-Critic Values:\n")
    print(format_values(values, env))

    print("\nActor-Critic Policy:\n")
    print(format_policy(policy, env))

    agent = PolicyGradientAgent(env)
    iters, values, policy = agent.train()

    print(f"\nPolicy Gradient (REINFORCE) converged in {iters} iterations.\n")

    print("\nPolicy Gradient Values:\n")
    print(format_values(values, env))

    print("\nPolicy Gradient Policy:\n")
    print(format_policy(policy, env))


if __name__ == "__main__":
    main()
