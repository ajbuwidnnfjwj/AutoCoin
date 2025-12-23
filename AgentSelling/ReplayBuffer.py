import torch
from collections import deque
import random

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_price, state_balance, action, reward, next_price, next_balance, done):
        sp = torch.as_tensor(state_price, dtype=torch.float32)
        sb = torch.as_tensor(state_balance, dtype=torch.float32)
        np_ = torch.as_tensor(next_price, dtype=torch.float32)
        nb = torch.as_tensor(next_balance, dtype=torch.float32)
        a = int(action)
        r = float(reward)
        d = float(done)
        self.buffer.append((sp, sb, a, r, np_, nb, d))

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