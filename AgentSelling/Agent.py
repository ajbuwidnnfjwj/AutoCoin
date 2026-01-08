import copy

import torch
import torch.nn as nn
import torch.optim as optim
import random

import pyupbit

# DQN Agent
class Agent:
    def __init__(self, model, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device='cpu',
                 num_actions=3, min_buy_cash=5000, train: bool=True):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = device
        self.target_model.to(self.device)
        self.model.to(self.device)

        self.num_actions = num_actions
        self.min_buy_cash = float(min_buy_cash)

        self.upbit = pyupbit.Upbit("my access key", "my secret key") if not train else None

    @torch.no_grad()
    def act(self, price_seq, balance):
        valid_actions = self._valid_actions(balance, self.min_buy_cash)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        

        price = torch.as_tensor(price_seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,T,M]
        bal = torch.as_tensor(balance, dtype=torch.float32, device=self.device).unsqueeze(0)      # [1,2]

        q_vals = self.model(price, bal)  # [1,A]

        cash = float(balance[0])
        coin = float(balance[1])

        if 1 not in valid_actions:
            q_vals[0, 1] = -1e9
        if 2 not in valid_actions:
            q_vals[0, 2] = -1e9

        return int(torch.argmax(q_vals, dim=1).item())

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        
        s_p, s_b, a, r, n_p, n_b, d = replay_buffer.sample(batch_size)

        s_p = s_p.to(self.device)
        s_b = s_b.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        n_p = n_p.to(self.device)
        n_b = n_b.to(self.device)
        d = d.to(self.device)

        # Q(s,a)
        q_all = self.model(s_p, s_b)                          # [B,A]
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)     # [B]

        with torch.no_grad():
            # DQN target: r + gamma*max_a Q_target(s',a)
            next_q = self.target_model(n_p, n_b)              # [B,A]

            mask = torch.zeros_like(next_q)
            mask[(n_b[:, 0] < self.min_buy_cash), 1] = -1e9   # buy 금지
            mask[(n_b[:, 1] <= 0.0), 2] = -1e9                # sell 금지
            next_q = next_q + mask

            target = r + self.gamma * next_q.max(1).values * (1.0 - d)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @torch.no_grad()
    def predict(self, state):
        price_seq, balance = state
        price = torch.as_tensor(price_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        bal = torch.as_tensor(balance, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.model(price, bal)

        cash = float(balance[0])
        coin = float(balance[1])
        if cash < self.min_buy_cash:
            q_vals[0, 1] = -1e9
        if coin <= 0.0:
            q_vals[0, 2] = -1e9
        return int(torch.argmax(q_vals, dim=1).item())

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _valid_actions(self, balance, min_buy_cash=10_000.0):
        cash = float(balance[0])
        coin = float(balance[1])

        valid = [0]  # hold는 항상 가능
        if cash >= min_buy_cash:
            valid.append(1)  # buy
        if coin > 0.0:
            valid.append(2)  # sell
        return valid