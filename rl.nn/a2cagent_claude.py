"""A2CAgent: Advantage Actor-Critic agent implementation for discrete action spaces."""

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


class A2CAgent:
    """Advantage Actor-Critic agent using TD bootstrapping, entropy bonus, and a shared optimizer."""

    SEED = 42
    NUM_EPISODES = 1000
    HIDDEN_DIM = 128
    GAMMA = 0.99
    LR = 5e-3
    ENTROPY_COEF = 0.01  # weight for entropy bonus — encourages exploration
    VALUE_COEF = 0.5  # weight for value loss relative to policy loss
    MAX_GRAD_NORM = 0.5  # gradient clipping threshold
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

        # Single optimizer for both networks — standard in A2C
        self._optimizer = torch.optim.AdamW(
            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
            lr=self.LR,
        )

        self._reward_history: list[float] = []

    def train(self) -> None:
        """Main training loop for the A2C agent."""

        start_time = time.monotonic()

        for episode in range(self.NUM_EPISODES):
            states, actions, next_states, rewards, dones = self._run_episode()
            self._update(states, actions, next_states, rewards, dones)

            total_reward = sum(rewards)
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
                "agent": "A2C",
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

    def _select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        logits = self._policy_net(state_tensor)
        dist = dists.Categorical(logits=logits)

        return dist.sample().item()

    def _run_episode(self) -> tuple[list, list, list, list, list]:
        states, actions, next_states, rewards, dones = [], [], [], [], []
        state, _ = self._env.reset()
        done = False

        while not done:
            action = self._select_action(state)
            next_state, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        return states, actions, next_states, rewards, dones

    def _compute_td_targets(self, next_states: list, rewards: list, dones: list) -> torch.Tensor:
        """TD target: r + γ·V(s') for non-terminal steps, r for terminal steps."""
        with torch.no_grad():
            next_states_tensor = torch.from_numpy(np.array(next_states)).float()
            next_values = self._value_net(next_states_tensor).squeeze(1)

        targets = [
            r if done else r + self.GAMMA * v_next.item() for r, v_next, done in zip(rewards, next_values, dones)
        ]

        return torch.tensor(targets, dtype=torch.float32)

    def _update(self, states: list, actions: list, next_states: list, rewards: list, dones: list) -> None:
        states_tensor = torch.from_numpy(np.array(states)).float()
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        td_targets = self._compute_td_targets(next_states, rewards, dones)
        values = self._value_net(states_tensor).squeeze(1)

        # Advantage: TD error — how much better the outcome was vs the critic's estimate.
        # .detach() prevents advantage gradient from flowing into the value network.
        advantages = (td_targets - values).detach()

        # Recompute distributions from stored states for clean entropy calculation
        logits = self._policy_net(states_tensor)
        dist = dists.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)

        policy_loss = -(log_probs * advantages).sum()
        value_loss = nn.functional.mse_loss(values, td_targets)
        entropy_bonus = dist.entropy().mean()

        # Combined loss: actor + critic - entropy regularization
        total_loss = policy_loss + self.VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy_bonus

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
            self.MAX_GRAD_NORM,
        )
        self._optimizer.step()


if __name__ == "__main__":
    agent = A2CAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/a2c_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
