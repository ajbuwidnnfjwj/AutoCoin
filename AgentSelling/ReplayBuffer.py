import torch
from collections import deque
import random

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_price, state_balance, action, reward, next_price, next_balance, done):
        self.buffer.append((state_price, state_balance, action, reward, next_price, next_balance, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state_price, state_balance, action, reward, next_price, next_balance, done = zip(*batch)
        return (
            torch.stack(state_price),  # [B, T, M]
            torch.stack(state_balance),  # [B, 2]
            torch.tensor(action, dtype=torch.int64),  # [B]
            torch.tensor(reward, dtype=torch.float32),  # [B]
            torch.stack(next_price),  # [B, T, M]
            torch.stack(next_balance),  # [B, 2]
            torch.tensor(done, dtype=torch.float32)  # [B]
        )

    def __len__(self):
        return len(self.buffer)