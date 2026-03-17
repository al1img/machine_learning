"""PGBAgent: Policy Gradient with Baseline agent using REINFORCE + learned value baseline."""

import json
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as distr
import torch.nn as nn
from common import compute_returns, plot_result


class PolicyNetwork(nn.Module):
    """Feedforward neural network for approximating policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # nn.Softmax(dim=-1) can be applied here but we will use Categorical distribution with logits, so no need to
            # apply softmax here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities given the input state."""
        return self._net(x)


class ValueNetwork(nn.Module):
    """Feedforward neural network for approximating state value function (baseline)."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """Returns state value given the input state."""
        return self._net(x)


class PGBAgent:
    """Policy Gradient (PG) agent for discrete action spaces using REINFORCE algorithm."""

    SEED = 42
    NUM_EPISODES = 1000
    HIDDEN_DIM = 128
    GAMMA = 0.99
    LR = 5e-3
    SOLVE_THRESHOLD = 475.0
    SOLVE_WINDOW = 100

    def __init__(self, env: gym.Env):
        self._env = env

        if self.SEED is not None:
            random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            self._env.reset(seed=self.SEED)
            self._env.action_space.seed(self.SEED)
            self._env.observation_space.seed(self.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self._policy_net = PolicyNetwork(state_dim, action_dim, self.HIDDEN_DIM)
        self._value_net = ValueNetwork(state_dim, self.HIDDEN_DIM)
        # self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self.LR)
        # Adam without amsgrad is safer for REINFORCE (amsgrad=False)
        self._policy_optimizer = torch.optim.AdamW(self._policy_net.parameters(), lr=self.LR)
        self._value_optimizer = torch.optim.AdamW(self._value_net.parameters(), lr=self.LR)
        self._reward_history: list[float] = []

    def train(self) -> None:
        """Main training loop for the PG agent."""

        start_time = time.monotonic()
        self._policy_net.train()
        self._value_net.train()

        for episode in range(self.NUM_EPISODES):
            states, rewards, log_probs = self._run_episode()
            returns = compute_returns(rewards, self.GAMMA)

            # Normalize for variance reduction is not needed here because we will use a learned value baseline,
            # so we will just subtract the baseline from returns

            self._update(states, log_probs, returns)

            total_reward = sum(rewards)
            self._reward_history.append(total_reward)

            print(f"Episode {episode + 1}: reward = {total_reward:.1f}")

            window = self._reward_history[-self.SOLVE_WINDOW :]

            if len(window) == self.SOLVE_WINDOW and np.mean(window) >= self.SOLVE_THRESHOLD:
                break

        print(f"Training completed in {time.monotonic() - start_time:.2f} seconds at episode {episode + 1}.")

        self._env.close()

    def save_result(self, file_name: str) -> None:
        """Saves the training result (rewards and epsilon history) to a file."""
        results = {
            "params": {
                "agent": "My PG with baseline",
                "policy_optimizer": self._policy_optimizer.__class__.__name__,
                "value_optimizer": self._value_optimizer.__class__.__name__,
                "seed": self.SEED,
                "GAMMA": self.GAMMA,
                "LR": self.LR,
            },
            "rewards": self._reward_history,
        }

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def play(self, num_episodes: int = 1) -> None:
        """Runs the trained policy visually using the render_mode='human' environment."""
        env = gym.make(self._env.spec.id, render_mode="human")
        self._policy_net.eval()

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    action = int(self._policy_net(state_tensor).argmax(dim=1).item())

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"Play episode {episode + 1}: reward = {total_reward:.1f}")

        env.close()

    def _select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        logits = self._policy_net(state_tensor)
        dist = distr.Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def _run_episode(self) -> tuple[list[np.ndarray], list[float], list[torch.Tensor]]:
        states, rewards, log_probs = [], [], []
        done = False
        state, _ = self._env.reset()

        while not done:
            action, log_prob = self._select_action(state)
            states.append(state)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = self._env.step(action)
            rewards.append(reward)

            done = terminated or truncated

        return states, rewards, log_probs

    def _update(self, states: list[np.ndarray], log_probs: list[torch.Tensor], returns: torch.Tensor):
        states_tensor = torch.from_numpy(np.array(states)).float()
        values = self._value_net(states_tensor).squeeze(1)

        # Advantage: how much better the actual return was vs the baseline estimate.
        # .detach() ensures advantage gradient does not flow into the value network.
        advantages = returns - values.detach()

        policy_loss = torch.stack([-lp * G for lp, G in zip(log_probs, advantages)]).sum()

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        value_loss = nn.functional.mse_loss(values, returns)
        self._value_optimizer.zero_grad()
        value_loss.backward()
        self._value_optimizer.step()


if __name__ == "__main__":
    agent = PGBAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/pgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
