"""PGAgent: Policy Gradient (PG) agent implementation for discrete action spaces."""

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


class PGAgent:
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
        # self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self.LR)
        # Adam without amsgrad is safer for REINFORCE (amsgrad=False)
        self._optimizer = torch.optim.AdamW(self._policy_net.parameters(), lr=self.LR)
        self._reward_history: list[float] = []

    def train(self) -> None:
        """Main training loop for the PG agent."""

        start_time = time.monotonic()
        self._policy_net.train()

        for episode in range(self.NUM_EPISODES):
            rewards, log_probs = self._run_episode()
            returns = compute_returns(rewards, self.GAMMA)

            # Normalize for variance reduction. This actually, average returns as baseline plus scaling by std,
            # is a common trick to stabilize training.
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            self._update(log_probs, returns)

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
                "agent": "My PG`",
                "optimizer": self._optimizer.__class__.__name__,
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

    def _run_episode(self) -> tuple[list[float], list[torch.Tensor]]:
        rewards, log_probs = [], []
        done = False
        state, _ = self._env.reset()

        while not done:
            action, log_prob = self._select_action(state)
            log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = self._env.step(action)
            rewards.append(reward)

            done = terminated or truncated

        return rewards, log_probs

    def _update(self, log_probs: list[torch.Tensor], returns: torch.Tensor):
        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


if __name__ == "__main__":
    agent = PGAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
