import torch

from AgentSelling.Env import Env
from AgentSelling.Model import TransformerEncoder
from AgentSelling.ReplayBuffer import ReplayBuffer
from AgentSelling.Agent import Agent
from Config import *
from log import Logger

train_logger = Logger('train', path=TRAIN_LOG_PATH)

# Training loop sketch
def train(agent, env, replay_buffer, num_episodes=1000,
          batch_size=64, target_update_freq=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else device)
    print(device)
    for ep in range(1, num_episodes + 1):
        price_seq, balances = env.reset()  # returns [T, M], [2]
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(price_seq, balances)
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

        msg = f"Episode {ep}, Reward: {ep_reward:.2f}, Epsilon: {agent.epsilon:.3f}"
        train_logger.logger.info(msg)
    torch.save(agent.model.state_dict(), MODEL_PARAM_PATH)
    torch.save(agent.target_model.state_dict(), TMODEL_PARAM_PATH)

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