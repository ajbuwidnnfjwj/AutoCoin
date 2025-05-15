import pyupbit
import numpy as np
from Config import access, secret

class Env:
    def __init__(self, window_size=10):
        self.window = window_size
        self.market = np.array(pyupbit.get_ohlcv("KRW-BTC").reset_index()[[
            'open', 'high', 'low', 'close', 'volume'
        ]])
        self.upbit = pyupbit.Upbit(access, secret)
        self.init_cash = self.upbit.get_balance("KRW")
        self.init_coin = self.upbit.get_balance("BTC")
        self.cash = self.init_cash
        self.coin = self.init_coin
        self.t = self.window

    def reset(self):
        self.t = self.window
        self.cash = self.init_cash
        self.coin = self.init_coin
        return self._get_state()

    def _get_state(self):
        # 1. 과거 window 크기의 가격 지표 벡터 (예: 종가만)
        price_window = self.market[self.t - self.window:self.t]
        # 2. 잔고 정보
        balance = np.array([self.cash, self.coin], dtype=np.float32)
        # 3. 결합
        return np.array(price_window, dtype=np.float32), balance

    def step(self, action):
        price = self.market[self.t][3]

        # 예시: action 0=Hold, 1=Buy, 2=Sell
        if action == 1:  # Buy all-in
            self.coin += self.cash / price
            self.cash = 0.0
        elif action == 2:  # Sell all
            self.cash += self.coin * price
            self.coin = 0.0
        # 보상 계산
        new_value = self.cash + self.coin * price
        old_value = self.cash + self.coin * self.market[self.t - 1][3]
        reward = (new_value - old_value) / (old_value + 1e-8)
        self.t += 1
        done = (self.t >= len(self.market))
        return *self._get_state() ,reward, done, {}

if __name__ == "__main__":
    env = Env()
    env.reset()
    env.step(1)