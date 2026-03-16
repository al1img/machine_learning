import random

import gym
import torch
import torch.nn as nn
import torch.optim as optim


# Define Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Initialize environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize Q-network and target network
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Initialize replay memory buffer
replay_buffer = []
buffer_capacity = 10000

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.99
batch_size = 64
update_target_interval = 100

# Optimizer
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Epsilon-greedy exploration strategy
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(500):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()  # Exploit

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > buffer_capacity:
            replay_buffer.pop(0)  # Remove oldest experience

        state = next_state

        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            for s, a, r, ns, d in minibatch:
                with torch.no_grad():
                    target_q_values = target_network(torch.tensor(ns, dtype=torch.float32))
                target = r if d else r + discount_factor * torch.max(target_q_values).item()
                predicted = q_network(torch.tensor(s, dtype=torch.float32))[a]
                loss = nn.MSELoss()(predicted, torch.tensor(target))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # target network updated once in a while .
        if episode % update_target_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()
