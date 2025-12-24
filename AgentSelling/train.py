import torch

from AgentSelling.Env import Env
from AgentSelling.Model import AgentModel, TransformerEncoder, PortfolioNet, PolicyNet
from AgentSelling.ReplayBuffer import ReplayBuffer
from AgentSelling.Agent import Agent
from log import Logger

import matplotlib.pyplot as plt
import numpy as np

import os
import copy

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
    plt.savefig(f'train_plot/episode_{episode}_chart.png')
    # plt.show() # Jupyer Notebook 등이 아닌 환경에서는 창이 멈출 수 있음
    plt.close()

# Training loop sketch
def train(agent, num_episodes=1000,
          batch_size=64, target_update_freq=10, replay_buff_capacity=10000, device='cpu',
          checkpoint :int = 100, save_best: bool = True,
          run_name: str = "default_run"):
    
    print(device)

    os.makedirs("model_params", exist_ok=True)
    os.makedirs("train_plot", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = Env(train=True)
    replay_buffer = ReplayBuffer(capacity=replay_buff_capacity)


    best_reward = -float('inf')
    best_param = None
    best_ep = 0
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

            next_price_seq, next_balances, reward, done = env.step(action)
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

        if ep % checkpoint == 0:
            torch.save(agent.model.state_dict(), f"model_params/{run_name}/model_ep_{ep}.pt")
            torch.save(agent.target_model.state_dict(), f"model_params/{run_name}/target_model_ep_{ep}.pt")

        if save_best and best_reward < ep_reward:
            best_reward = ep_reward
            best_param = copy.deepcopy(agent.model.state_dict())
            best_ep = ep
            msg = f"Epoch {ep}: New best model saved with reward {best_reward:.2f}"

    torch.save(agent.model.state_dict(), f"model_params/{run_name}/model.pt")
    torch.save(agent.target_model.state_dict(), f"model_params/{run_name}/target_model.pt")
    if save_best and best_param is not None:
        torch.save(best_param, f"model_params/{run_name}/best_model_epoch_{best_ep}.pt")
        train_logger.logger.info(msg)

    return {"best_reward": best_reward, "best_ckpt": f"model_params/{run_name}/best_model_epoch_{best_ep}.pt"}

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else device)

    model = AgentModel(
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

    env = Env(train=True)
    replay_buffer = ReplayBuffer(capacity=10000)
    agent = Agent(model=model, lr=1e-4, gamma=0.99, device=device, train=True)
    train(agent, env, replay_buffer, num_episodes=1000, device=device)