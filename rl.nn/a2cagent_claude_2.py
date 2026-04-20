"""A2CAgent2: Strict online TD(0) Advantage Actor-Critic — updates at every step, not per episode."""

import json
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
from common import plot_result


class PolicyNetwork(nn.Module):
    """Feedforward neural network for approximating policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class ValueNetwork(nn.Module):
    """Feedforward neural network for approximating state value V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class A2CAgent2:
    """Strict online TD(0) A2C — networks are updated at every environment step."""

    SEED = 42
    NUM_EPISODES = 1000
    HIDDEN_DIM = 128
    GAMMA = 0.99
    LR = 5e-3
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
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

        self._optimizer = torch.optim.AdamW(
            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
            lr=self.LR,
        )

        self._reward_history: list[float] = []

    def train(self) -> None:
        """Main training loop — each episode runs steps and updates online."""

        start_time = time.monotonic()

        for episode in range(self.NUM_EPISODES):
            total_reward = self._run_episode()
            self._reward_history.append(total_reward)

            print(f"Episode {episode + 1}: reward = {total_reward:.1f}")

            window = self._reward_history[-self.SOLVE_WINDOW :]
            if len(window) == self.SOLVE_WINDOW and np.mean(window) >= self.SOLVE_THRESHOLD:
                break

        print(f"Training completed in {time.monotonic() - start_time:.2f} seconds at episode {episode + 1}.")

        self._env.close()

    def save_result(self, file_name: str) -> None:
        """Saves the training result to a file."""
        results = {
            "params": {
                "agent": "A2C2",
                "optimizer": self._optimizer.__class__.__name__,
                "seed": self.SEED,
                "GAMMA": self.GAMMA,
                "LR": self.LR,
                "ENTROPY_COEF": self.ENTROPY_COEF,
                "VALUE_COEF": self.VALUE_COEF,
            },
            "rewards": self._reward_history,
        }

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def play(self, num_episodes: int = 1) -> None:
        """Runs the trained policy visually using the render_mode='human' environment."""
        env = gym.make(self._env.spec.id, render_mode="human")
        self._policy_net.eval()
        self._value_net.eval()

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
        self._policy_net.train()
        self._value_net.train()

    def _run_episode(self) -> float:
        """Runs one episode, updating the networks at every step. Returns total reward."""
        state, _ = self._env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

            # Actor: sample action and get log prob
            logits = self._policy_net(state_tensor)
            dist = dists.Categorical(logits=logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()

            # Critic: estimate V(s)
            value = self._value_net(state_tensor).squeeze(1)

            next_state, reward, terminated, truncated, _ = self._env.step(action_tensor.item())
            done = terminated or truncated
            total_reward += reward

            # TD target: bootstrap from V(s') unless terminal
            with torch.no_grad():
                if done:
                    td_target = torch.tensor([reward], dtype=torch.float32)
                else:
                    next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
                    td_target = reward + self.GAMMA * self._value_net(next_state_tensor).squeeze(1)

            # Advantage: TD error δ = r + γV(s') - V(s)
            advantage = (td_target - value).detach()

            policy_loss = -(log_prob * advantage)
            value_loss = nn.functional.mse_loss(value, td_target)
            total_loss = policy_loss + self.VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy

            self._optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._policy_net.parameters()) + list(self._value_net.parameters()),
                self.MAX_GRAD_NORM,
            )
            self._optimizer.step()

            state = next_state

        return total_reward


if __name__ == "__main__":
    agent = A2CAgent2(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/a2c2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
