import pyupbit
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Env
from Model import TransformerEncoder
from Agent import Agent

model = TransformerEncoder(
        input_dim=7,  # ohlcv
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_ff=256,
        num_actions=3,  # Buy / Sell / Hold
        max_len=100,
        dropout=0.1
)
model.load_state_dict(torch.load('test_model_param/target_model.pt'))
agent = Agent(model=model, init_args=(7,64,4,2,256,3,100,0.1), lr=1e-4, gamma=0.99)
env = Env.Env()
state = env.reset()
done = False

portfolio_values = []
timestamps       = []


price_series = np.array(pyupbit.get_ohlcv("KRW-BTC", interval="minute60").reset_index()[[
            'open', 'high', 'low', 'close', 'volume'
        ]])
data_series = pyupbit.get_ohlcv("KRW-BTC", interval="day200").index

while not done:
    # ε=0 으로 고정하여 완전 탐욕 정책으로 실행
    action = agent.predict(state, epsilon=0.0)
    next_price_seq, next_balances, reward, done, _ = env.step(action)
    state = (next_price_seq, next_balances)

    # 포트폴리오 현재 가치
    # env.cash, env.coin, 현재가격(env.t-1 인덱스) 사용 예시
    current_price      = price_series[env.t - 1][3]
    current_value      = env.cash + env.coin * current_price
    portfolio_values.append(current_value)

    # 날짜 기록 (env가 날짜를 관리하지 않으면 date_series[env.t-1] 사용)
    timestamps.append(data_series[env.t - 1])

# 3) 일별 수익률 계산
# pandas Series로 변환 후 pct_change() 이용
portfolio_series = pd.Series(portfolio_values, index=pd.to_datetime(timestamps))
daily_returns     = portfolio_series.pct_change().dropna()
cumulative_growth = (1 + daily_returns).cumprod()
net_cum_returns   = cumulative_growth - 1


plt.figure(figsize=(10, 6))
plt.plot(net_cum_returns)        # 또는 plt.plot(cumulative_growth) 로 성장지수 표시 가능
plt.title('누적 수익률 (Cumulative Returns)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()