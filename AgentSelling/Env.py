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
        '''action 0=Hold, 1=Buy, 2=Sell'''
        price = self.market[self.t][3]  # 현재 시점 종가
        prev_price = self.market[self.t - 1][3] if self.t > 0 else price
        old_value = self.cash + self.coin * prev_price

        next_cash = self.cash
        next_coin = self.coin
        reward = 0.0

        if (action == 1 and self.cash >= 10000) or (action == 2 and self.coin > 0.0):
            # 정상매매 루트
            if action == 1:
                krw_volume = self.cash - 5000
                next_coin = self.coin + krw_volume / price
                next_cash = self.cash - krw_volume
            elif action == 2:
                next_cash = self.cash + self.coin * price
                next_coin = 0.0

            # 보상 계산
            new_value = next_cash + next_coin * price
            reward = (new_value - old_value) / (old_value + 1e-8)
        elif (action == 1 and self.cash < 10000) or (action == 2 and self.coin <= 0.0):
            # 잔고와 맞지 않는 거래
            reward = -1

        self.cash = next_cash
        self.coin = next_coin
        self.t += 1
        done = (self.t >= len(self.market))
        return *self._get_state(),reward, done, {}

if __name__ == "__main__":
    env = Env()
    env.reset()
    env.step(1)