import torch
import torch.nn as nn
import torch.optim as optim
import random

import pyupbit
from Config import *

# DQN Agent
class Agent:
    def __init__(self, model, init_args, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device='cpu'):
        self.model = model
        self.target_model = type(model)(*init_args)  # same constructor args
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = device
        self.target_model.to(self.device)
        self.model.to(self.device)

        self.upbit = pyupbit.Upbit(access, secret)

    def act(self, price_seq, balance):
        if random.random() < self.epsilon:
            return random.randrange(self.model.head.out_features)
        price = torch.FloatTensor(price_seq).unsqueeze(0).to(self.device)  # [1, T, M]
        bal = torch.FloatTensor(balance).unsqueeze(0).to(self.device)  # [1, 2]
        with torch.no_grad():
            q_vals = self.model(price, bal)  # [1, num_actions]

        if balance[0] < 10000:
            q_vals[0][1] = -1e9
        if balance[1] <= 0:
            q_vals[0][2] = -1e9

        return int(q_vals.argmax(dim=1).item())

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        s_p, s_b, a, r, n_p, n_b, d = [t.to(self.device) for t in replay_buffer.sample(batch_size)]
        # Compute Q(s,a)
        q_values = self.model(s_p.to(self.device), s_b.to(self.device)).gather(1, a.unsqueeze(1)).squeeze(1)
        # Compute target Q
        with torch.no_grad():
            next_q = self.target_model(n_p.to(self.device), n_b.to(self.device)).max(1).values
            target = r + self.gamma * next_q * (1 - d)
        # Loss and optimize
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def predict(self, state, epsilon=0.0):
        price = torch.FloatTensor(state[0]).unsqueeze(0)
        bal = torch.FloatTensor(state[1]).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(price, bal)
        return int(q_vals.argmax(dim=1).item())

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())