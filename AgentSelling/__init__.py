import torch
import numpy as np

from AgentSelling.Model import AgentModel, TransformerEncoder, PortfolioNet, PolicyNet
from AgentSelling.Agent import Agent
from AgentSelling.Env import Env
from AgentSelling.train import train
from AgentSelling.ReplayBuffer import ReplayBuffer
from AgentSelling.log import Logger

import pyupbit

from dotenv import load_dotenv
import os

load_dotenv()



def build_model():
    return AgentModel(
        market_encoder=TransformerEncoder(
            ohlcv_dim=5,
            d_model=64,
            num_heads=4,
            num_layers=2,
            dim_ff=256,
            max_len=100,
            dropout=0.1
        ),
        balance_net=PortfolioNet(portfolio_dim=2, hidden_dim=32, out_dim=64),
        policy_net=PolicyNet(encode_dim=128, hidden_dim=32, num_actions=3)
    )

upbit = pyupbit.Upbit(access=os.getenv("access"), secret=os.getenv("secrete"))
logger = Logger("AgentSelling")

def RunAgentSell(retrain_on_traid=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else device)

    model = build_model()
    agent = Agent(model=model, lr=1e-4, gamma=0.99, device=device, train=False)
    try:
        model.load_state_dict(torch.load("model_params/model.pt"))
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

        if retrain_on_traid:
            env = Env(train=True)
            replay_buffer = ReplayBuffer(capacity=10000)
            train(agent, env, replay_buffer, num_episodes=1000)

    except Exception as e:
        logger.logger.error(e)