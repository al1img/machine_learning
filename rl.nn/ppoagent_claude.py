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


class PolicyNetwork(nn.Module):
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


class ValueNetwork(nn.Module):
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
        self._policy_net = PolicyNetwork(state_dim, action_dim, self.HIDDEN_DIM)
        self._value_net = ValueNetwork(state_dim, self.HIDDEN_DIM)
        #        self._optimizer = torch.optim.AdamW(
        #            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
        #            lr=self.LR,
        #            eps=1e-5,
        #        )
        self._optimizer = torch.optim.Adam(
            list(self._policy_net.parameters()) + list(self._value_net.parameters()),
            lr=self.LR,
            eps=1e-5,
        )
        self._reward_history: list[float] = []

    def train(self, total_steps: int = 500_000) -> None:
        """Main training loop. Collects rollouts and updates the policy until total_steps."""

        start_time = time.monotonic()
        self._policy_net.train()
        self._value_net.train()

        steps_done = 0
        episode = 0
        state, _ = self._env.reset()
        episode_reward = 0.0

        while steps_done < total_steps:
            # --- collect rollout ---
            rollout = self._collect_rollout(state, episode_reward)
            state = rollout["next_state"]
            episode_reward = rollout["next_episode_reward"]
            steps_done += rollout["steps"]

            for ep_reward in rollout["completed_rewards"]:
                episode += 1
                self._reward_history.append(ep_reward)
                print(f"Episode {episode}: reward = {ep_reward:.1f}  steps = {steps_done}")

                window = self._reward_history[-self.SOLVE_WINDOW :]
                if len(window) == self.SOLVE_WINDOW and np.mean(window) >= self.SOLVE_THRESHOLD:
                    print(f"Solved at episode {episode}.")
                    self._env.close()
                    print(f"Training completed in {time.monotonic() - start_time:.2f} seconds.")
                    return

            # --- PPO update ---
            self._update(rollout)

        print(f"Training completed in {time.monotonic() - start_time:.2f} seconds at step {steps_done}.")
        self._env.close()

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_rollout(self, init_state: np.ndarray, init_episode_reward: float) -> dict:
        """Collects ROLLOUT_STEPS transitions from the environment.

        Returns a dict with tensors needed for the PPO update plus bookkeeping.
        """
        states, actions, rewards, dones, log_probs_old, values = [], [], [], [], [], []
        completed_rewards: list[float] = []

        state = init_state
        episode_reward = init_episode_reward

        self._policy_net.eval()
        self._value_net.eval()

        with torch.no_grad():
            for _ in range(self.ROLLOUT_STEPS):
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                logits = self._policy_net(state_t)
                dist = dists.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self._value_net(state_t).squeeze()

                next_state, reward, terminated, truncated, _ = self._env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action.item())
                rewards.append(float(reward))
                dones.append(float(done))
                log_probs_old.append(log_prob.item())
                values.append(value.item())

                episode_reward += float(reward)
                state = next_state

                if done:
                    completed_rewards.append(episode_reward)
                    episode_reward = 0.0
                    state, _ = self._env.reset()

            # Bootstrap value for the last state (needed for GAE)
            last_value = 0.0 if dones[-1] else self._value_net(torch.from_numpy(state).float().unsqueeze(0)).item()

        advantages = self._compute_gae(rewards, values, dones, last_value)
        returns = advantages + torch.tensor(values, dtype=torch.float32)

        return {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "log_probs_old": torch.tensor(log_probs_old, dtype=torch.float32),
            "advantages": advantages,
            "returns": returns,
            "steps": self.ROLLOUT_STEPS,
            "next_state": state,
            "next_episode_reward": episode_reward,
            "completed_rewards": completed_rewards,
        }

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[float],
        last_value: float,
    ) -> torch.Tensor:
        """Generalized Advantage Estimation (GAE-lambda)."""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + self.GAMMA * self.GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae

        return torch.from_numpy(advantages)

    def _update(self, rollout: dict) -> None:
        """Performs EPOCHS passes of minibatch PPO updates on the collected rollout."""
        states = rollout["states"]
        actions = rollout["actions"]
        log_probs_old = rollout["log_probs_old"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]

        # Normalize advantages over the whole rollout buffer
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        self._policy_net.train()
        self._value_net.train()

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
                logits = self._policy_net(mb_states)
                dist = dists.Categorical(logits=logits)
                mb_log_probs_new = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate loss
                ratio = torch.exp(mb_log_probs_new - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to stabilize)
                mb_values = self._value_net(mb_states).squeeze(1)
                value_loss = nn.functional.mse_loss(mb_values, mb_returns)

                loss = policy_loss + self.VALUE_COEF * value_loss - self.ENTROPY_COEF * entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._policy_net.parameters()) + list(self._value_net.parameters()),
                    self.MAX_GRAD_NORM,
                )
                self._optimizer.step()


if __name__ == "__main__":
    agent = PPOAgent(gym.make("CartPole-v1"))
    agent.train()

    filename = f"results/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    agent.save_result(filename)

    print(f"Saved results to {filename}")

    plot_result(filename)

    agent.play()
