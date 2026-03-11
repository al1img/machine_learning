import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from common import plot_result


@dataclass(frozen=True)
class EpisodeItem:
    """Represents an item in an episode."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Replay buffer for storing and sampling episode items."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, item: EpisodeItem):
        """Adds an episode item to the buffer."""
        self.buffer.append(item)

    def sample(self, batch_size: int):
        """Samples a batch of episode items and returns tensors for training."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Feedforward neural network for approximating Q-values."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions given the input state."""
        return self._net(x)


class DQNAgent:
    """Deep Q-Network (DQN) agent for training on discrete action spaces."""

    SEED = 42

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.98

    NUM_EPISODES = 500
    BUFFER_SIZE = 10000

    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    GAMMA = 0.99
    TAU = 0.005  # soft update coefficient: θ' ← τ·θ + (1-τ)·θ'
    LR = 1e-3

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
        self._policy_net = QNetwork(state_dim, action_dim, self.HIDDEN_DIM)
        self._target_net = QNetwork(state_dim, action_dim, self.HIDDEN_DIM)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()  # Target network is not trained directly
        self._buffer = ReplayBuffer(self.BUFFER_SIZE)
        # self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self.LR)
        self._optimizer = torch.optim.AdamW(self._policy_net.parameters(), lr=self.LR, amsgrad=True)
        self._reward_history: list[float] = []
        self._epsilon_history: list[float] = []
        self._current_episode = 0

    def train(self) -> None:
        """Main training loop for the DQN agent."""

        epsilon = self.EPSILON_START
        start_time = time.monotonic()

        for episode in range(self.NUM_EPISODES):
            self._current_episode = episode
            state, _ = self._env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action = self._select_action(state, epsilon)

                next_state, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                # unclear shall done or terminated be used here: in PyTorch example they use terminated
                self._buffer.push(EpisodeItem(state, action, reward, next_state, terminated))
                self._train_step()
                self._update_target()

                state = next_state
                episode_reward += reward

            self._reward_history.append(episode_reward)
            self._epsilon_history.append(epsilon)
            epsilon = max(self.EPSILON_END, epsilon * self.EPSILON_DECAY)

            print(f"Episode {episode + 1}: reward = {episode_reward:.1f}, epsilon = {epsilon:.3f}")

        print(f"Training completed in {time.monotonic() - start_time:.2f} seconds")

        self._env.close()

    def save_result(self, file_name: str) -> None:
        """Saves the training result (rewards and epsilon history) to a file."""
        results = {
            "params": {
                "agent": "My DQN",
                "optimizer": self._optimizer.__class__.__name__,
                "seed": self.SEED,
                "BATCH_SIZE": self.BATCH_SIZE,
                "GAMMA": self.GAMMA,
                "EPS_START": self.EPSILON_START,
                "EPS_END": self.EPSILON_END,
                "EPS_DECAY": self.EPSILON_DECAY,
                "TAU": self.TAU,
                "LR": self.LR,
            },
            "rewards": self._reward_history,
            "epsilons": self._epsilon_history,
        }

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() > epsilon:
            with torch.no_grad():
                # Convert state to tensor and add batch dimension
                x = torch.from_numpy(state).float().unsqueeze(0)

                return int(self._policy_net(x).argmax(dim=1).item())

        return self._env.action_space.sample()

    def _train_step(self) -> None:
        if len(self._buffer) < self.BATCH_SIZE:
            return

        batch = self._buffer.sample(self.BATCH_SIZE)

        states = torch.from_numpy(np.stack([s.state for s in batch])).float()
        actions = torch.from_numpy(np.array([s.action for s in batch])).long()
        rewards = torch.from_numpy(np.array([s.reward for s in batch])).float()
        next_states = torch.from_numpy(np.stack([s.next_state for s in batch])).float()
        dones = torch.from_numpy(np.array([s.done for s in batch])).float()

        # Current Q-values for taken actions
        q_values = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD targets using the frozen target network
        with torch.no_grad():
            next_q = self._target_net(next_states).max(dim=1).values
            targets = rewards + self.GAMMA * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()

    def _update_target(self) -> None:
        for policy_param, target_param in zip(self._policy_net.parameters(), self._target_net.parameters()):
            target_param.data.copy_(self.TAU * policy_param.data + (1.0 - self.TAU) * target_param.data)


if __name__ == "__main__":
    agent = DQNAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)
