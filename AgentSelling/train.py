import torch

from AgentSelling.Env import Env
from AgentSelling.Model import TransformerEncoder
from AgentSelling.ReplayBuffer import ReplayBuffer
from AgentSelling.Agent import Agent
from log import Logger

import matplotlib.pyplot as plt
import numpy as np

train_logger = Logger('train', path="logs/trainlog.log")


def visualize_episode(episode, history):
    prices = history['prices']
    actions = history['actions']
    balances = history['balances']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [3, 1]})

    fig.suptitle(f'Episode {episode} Trading History', fontsize=16)

    # --- 1번 차트: 가격 및 매매 시점 ---
    ax1.plot(prices, label='Price', color='gray')

    # 매수/매도 시점 찾기
    buy_points = np.where(np.array(actions) == 1)[0]
    sell_points = np.where(np.array(actions) == 2)[0]

    # 매수/매도 마커 표시
    if len(buy_points) > 0:
        ax1.scatter(buy_points, np.array(prices)[buy_points],
                    label='Buy', marker='^', color='red', s=100, zorder=3)
    if len(sell_points) > 0:
        ax1.scatter(sell_points, np.array(prices)[sell_points],
                    label='Sell', marker='v', color='blue', s=100, zorder=3)

    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # --- 2번 차트: 자산 변동 ---
    ax2.plot(balances, label='Total Asset', color='purple')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Asset Value')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 그래프를 파일로 저장하거나 직접 보기
    plt.savefig(f'episode_{episode}_chart.png')
    # plt.show() # Jupyer Notebook 등이 아닌 환경에서는 창이 멈출 수 있음
    plt.close()

# Training loop sketch
def train(agent, env, replay_buffer, num_episodes=1000,
          batch_size=64, target_update_freq=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else device)
    print(device)
    for ep in range(1, num_episodes + 1):
        episode_history = {
            'prices': [],
            'actions': [],
            'balances': []
        }

        price_seq, balances = env.reset()  # returns [T, M], [2]
        done = False
        ep_reward = 0.0
        actions = []
        while not done:
            action = agent.act(price_seq, balances)

            episode_history['prices'].append(price_seq[-1][3])
            episode_history['actions'].append(action)
            episode_history['balances'].append(balances[0] + balances[1] * price_seq[-1][3])

            next_price_seq, next_balances, reward, done, _ = env.step(action)
            replay_buffer.push(
                torch.FloatTensor(price_seq),
                torch.FloatTensor(balances),
                action,
                reward,
                torch.FloatTensor(next_price_seq),
                torch.FloatTensor(next_balances),
                done
            )
            agent.update(replay_buffer, batch_size)
            price_seq, balances = next_price_seq, next_balances
            ep_reward += reward

        # Update target network
        if ep % target_update_freq == 0:
            agent.update_target()

        visualize_episode(ep, episode_history)
        msg = f"Episode {ep}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
        train_logger.logger.info(msg)
    torch.save(agent.model.state_dict(), "model_params/model.pt")
    torch.save(agent.target_model.state_dict(), "model_params/target_model.pt")

# Example initialization (assuming you have env and model from earlier)
# env = TradingEnv(price_series, window_size=200)
# model = TradingAgentModel(input_dim=M, d_model=64, num_heads=4,
#                           num_layers=2, dim_ff=256, max_len=200,
#                           dropout=0.1, mlp_hidden=32, num_actions=3)
# model._init_args = (M, 64, 4, 2, 256, 200, 0.1, 32, 3)  # used for target model instantiation
# replay_buffer = ReplayBuffer(capacity=10000)
# agent = DQNAgent(model)
# train(agent, env, replay_buffer)

if __name__ == "__main__":
    env = Env()
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
    replay_buffer = ReplayBuffer(capacity=10000)
    agent = Agent(model=model, init_args=(7,64,4,2,256,3,100,0.1), lr=1e-4, gamma=0.99)
    train(agent, env, replay_buffer, num_episodes=1000)