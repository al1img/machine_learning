"""PPOAgent: Proximal Policy Optimization agent for discrete action spaces."""

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


class ActorNetwork(nn.Module):
    """Feedforward neural network for approximating policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits given the input state."""
        return self._net(x)


class CriticNetwork(nn.Module):
    """Feedforward neural network for approximating state value function (critic)."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns state value given the input state."""
        return self._net(x)


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent for discrete action spaces.

    Uses the clipped surrogate objective to constrain policy updates, preventing
    destructively large steps. Each rollout buffer is used for multiple epochs
    of minibatch gradient updates before being discarded.
    """

    SEED = 42
    NUM_EPISODES = 1000
    HIDDEN_DIM = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95  # GAE (Generalized Advantage Estimation) lambda
    CLIP_EPS = 0.2  # PPO clipping epsilon
    LR = 3e-4
    EPOCHS = 10  # number of gradient epochs per rollout
    BATCH_SIZE = 64  # minibatch size for each gradient update
    ROLLOUT_STEPS = 2048  # steps to collect per rollout
    ENTROPY_COEF = 0.01  # entropy bonus coefficient
    VALUE_COEF = 0.5  # value loss coefficient
    MAX_GRAD_NORM = 0.5  # gradient clipping norm
    SOLVE_THRESHOLD = 475.0
    SOLVE_WINDOW = 100

    def __init__(self, env: gym.Env):
        self._env = env

        if self.SEED is not None:
            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            self._env.reset(seed=self.SEED)
            self._env.action_space.seed(self.SEED)
            self._env.observation_space.seed(self.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self._actor_net = ActorNetwork(state_dim, action_dim, self.HIDDEN_DIM)
        self._critic_net = CriticNetwork(state_dim, self.HIDDEN_DIM)
        #        self._optimizer = torch.optim.AdamW(
        #            list(self._actor_net.parameters()) + list(self._critic_net.parameters()),
        #            lr=self.LR,
        #            eps=1e-5,
        #        )
        self._optimizer = torch.optim.Adam(
            list(self._actor_net.parameters()) + list(self._critic_net.parameters()),
            lr=self.LR,
            eps=1e-5,
        )
        self._next_state: np.ndarray = []
        self._next_done: bool = False
        self._next_value: float = 0.0
        self._episode_reward = 0.0
        self._reward_history: list[float] = []
        self._train_done = False
        self._episode_count = 0

    def train(self) -> None:
        """Main training loop. Collects rollouts and updates the policy until total_steps."""

        start_time = time.monotonic()

        self._reset()

        while True:
            states, actions, log_probs_old, advantages, returns = self._rollout()

            if self._train_done:
                break

            self._update(states, actions, log_probs_old, advantages, returns)

        print(f"Training completed in {time.monotonic() - start_time:.2f} seconds at episode {self._episode_count}.")

        self._env.close()

    def play(self, num_episodes: int = 1) -> None:
        """Runs the trained policy visually using the render_mode='human' environment."""
        env = gym.make(self._env.spec.id, render_mode="human")
        self._actor_net.eval()

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    action = int(self._actor_net(state_tensor).argmax(dim=1).item())

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            print(f"Play episode {episode + 1}: reward = {total_reward:.1f}")

        env.close()

    def save_result(self, file_name: str) -> None:
        """Saves the training result to a JSON file."""
        results = {
            "params": {
                "agent": "PPO",
                "optimizer": self._optimizer.__class__.__name__,
                "seed": self.SEED,
                "GAMMA": self.GAMMA,
                "GAE_LAMBDA": self.GAE_LAMBDA,
                "CLIP_EPS": self.CLIP_EPS,
                "LR": self.LR,
                "EPOCHS": self.EPOCHS,
                "BATCH_SIZE": self.BATCH_SIZE,
                "ROLLOUT_STEPS": self.ROLLOUT_STEPS,
            },
            "rewards": self._reward_history,
        }

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f)

    def _reset(self) -> None:
        """Resets the environment and internal state for a new episode."""
        self._next_state, _ = self._env.reset()
        self._next_done = False
        self._next_value = 0.0
        self._episode_reward = 0.0
        self._train_done = False

    def _rollout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collects a rollout of experience by interacting with the environment.

        Returns a tuple containing tensors for states, actions, log probabilities of actions, advantages, and returns.
        """
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        for _ in range(self.ROLLOUT_STEPS):
            state_tensor = torch.from_numpy(self._next_state).float().unsqueeze(0)

            with torch.no_grad():
                action_logits = self._actor_net(state_tensor)
                action_dist = dists.Categorical(logits=action_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            states.append(self._next_state)
            dones.append(self._next_done)
            actions.append(action.item())
            log_probs.append(log_prob)
            values.append(self._next_value)

            # next step

            next_state, reward, terminated, truncated, _ = self._env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)

            if done:
                next_state, _ = self._env.reset()
                self._episode_done(self._episode_reward)
                self._episode_reward = 0.0

                if self._train_done:
                    break

            self._episode_reward += reward
            self._next_state = next_state
            self._next_done = done

            with torch.no_grad():
                self._next_value = self._critic_net(torch.from_numpy(self._next_state).float().unsqueeze(0)).item()

        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(log_probs, dtype=torch.float32),
            advantages,
            returns,
        )

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[float],
    ) -> torch.Tensor:
        """Generalized Advantage Estimation (GAE-lambda)."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else self._next_value
            next_non_terminal = 1.0 - float(dones[t + 1] if t < len(rewards) - 1 else self._next_done)
            delta = rewards[t] + self.GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + self.GAMMA * self.GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae

        return torch.from_numpy(advantages)

    def _update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """Performs EPOCHS passes of minibatch PPO updates on the collected rollout."""

        self._actor_net.train()
        self._critic_net.train()

        # Normalize advantages over the whole rollout buffer
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)

        for _ in range(self.EPOCHS):
            indices = torch.randperm(n)

            for start in range(0, n, self.BATCH_SIZE):
                mb_idx = indices[start : start + self.BATCH_SIZE]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs_old = log_probs_old[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # New log-probs and entropy under current policy
                logits = self._actor_net(mb_states)
                dist = dists.Categorical(logits=logits)
                mb_log_probs_new = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate loss
                ratio = torch.exp(mb_log_probs_new - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to stabilize)
                mb_values = self._critic_net(mb_states).squeeze(1)
                value_loss = nn.functional.mse_loss(mb_values, mb_returns)

                loss = policy_loss + self.VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy

                self._optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    list(self._actor_net.parameters()) + list(self._critic_net.parameters()),
                    self.MAX_GRAD_NORM,
                )

                self._optimizer.step()

    def _episode_done(self, reward: float) -> None:
        """Handles end of episode logic, including reward tracking and environment reset."""
        self._reward_history.append(reward)

        self._episode_count += 1

        print(f"Episode {self._episode_count}: reward = {reward:.1f}")

        window = self._reward_history[-self.SOLVE_WINDOW :]

        if (
            len(window) == self.SOLVE_WINDOW
            and np.mean(window) >= self.SOLVE_THRESHOLD
            or self._episode_count >= self.NUM_EPISODES
        ):
            self._train_done = True


if __name__ == "__main__":
    agent = PPOAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
