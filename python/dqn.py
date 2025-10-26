"""
Double DQN Agent with PyTorch (LARGER CAPACITY)

Testing capacity hypothesis with wider backbone + LayerNorm
- Network: 144 → 256 → 128 → 64 → 6 (WITH LayerNorm)
- Double DQN (online + target networks)
- Experience replay buffer (capacity 20000)
- Polyak averaging for target network (τ=0.01)
- Adam optimizer (lr=0.005)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    """Q-Network: 144 → 256 → 128 → 64 → 6 (with LayerNorm)"""

    def __init__(self, state_dim=144, action_dim=6):
        super().__init__()
        # Wider backbone: 144→256→128→64→6
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)

        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)

        return self.fc4(x)  # No activation on output (Q-values can be any real number)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DoubleDQN:
    """Double DQN Agent"""

    def __init__(
        self,
        state_dim=144,
        action_dim=6,
        lr=0.005,
        gamma=0.99,
        tau=0.01,
        batch_size=64,
        buffer_capacity=20000,
        device=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Device
        if device is None:
            self.device = torch.device("cpu")  # CPU only for fair comparison
        else:
            self.device = device

        # Networks
        self.online_net = QNetwork(state_dim, action_dim=action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.online_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step (if enough samples in buffer)"""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values: Q(s, a)
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network (Polyak averaging)
        self._update_target_network()

        return loss.item()

    def _update_target_network(self):
        """Soft update: θ_target = τ * θ_online + (1 - τ) * θ_target"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )


if __name__ == "__main__":
    # Test DQN agent
    agent = DoubleDQN()
    print("Agent created successfully")

    # Test action selection
    state = np.random.randn(144).astype(np.float32)
    action = agent.select_action(state, epsilon=0.1)
    print(f"Selected action: {action}")

    # Test storing transitions
    next_state = np.random.randn(144).astype(np.float32)
    agent.store_transition(state, action, -0.01, next_state, False)
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
