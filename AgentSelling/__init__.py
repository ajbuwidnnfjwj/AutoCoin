import torch
import numpy as np

from AgentSelling.Model import TransformerEncoder
from AgentSelling.Env import Env
from AgentSelling.Agent import Agent
from AgentSelling.train import train
from AgentSelling.ReplayBuffer import ReplayBuffer
from log import Logger
from Config import *

import pyupbit

model = TransformerEncoder(
    input_dim=7,  # ohlcv
    d_model=64,
    num_heads=4,
    num_layers=2,
    dim_ff=256,
    num_actions=3,  # Buy / Sell / Hold
    max_len=100,
    dropout=0.1)
agent = Agent(model=model, init_args=(7, 64, 4, 2, 256, 3, 100, 0.1), lr=1e-4, gamma=0.99)
upbit = pyupbit.Upbit(access=access, secret=secret)
logger = Logger("AgentSelling")

def RunAgentSell():
    try:
        model.load_state_dict(torch.load(TMODEL_PARAM_PATH))
        prices = np.array(pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=10).reset_index()[[
                'open', 'high', 'low', 'close', 'volume'
            ]])
        balance = (upbit.get_balance("KRW"), upbit.get_balance("BTC"))
        action = agent.predict((prices, balance))
        logger.logger.info("hold" if action==0 else
                           "Buy" if action==1 else
                           "Sell" if action==2 else None)

        # action 0=Hold, 1=Buy, 2=Sell
        if action == 1:
            krw_balance = upbit.get_balance("KRW") - 5000
            if krw_balance < 5000:
                return
            upbit.buy_market_order("KRW-BTC", krw_balance)
        elif action == 2:
            btc_balance = upbit.get_balance("BTC")
            upbit.sell_market_order("KRW-BTC", btc_balance)

        env = Env()
        replay_buffer = ReplayBuffer(capacity=10000)
        train(agent, env, replay_buffer, num_episodes=1000)

    except Exception as e:
        logger.logger.error(e)