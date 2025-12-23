import torch

from AgentSelling.Env import Env
from AgentSelling.ReplayBuffer import ReplayBuffer
from AgentSelling.Agent import Agent
from AgentSelling.train import train

from AgentSelling import model, agent, device

if __name__ == "__main__":
    agent = Agent(model=model, lr=1e-4, gamma=0.99, device=device, train=True)
    train(agent, num_episodes=1000, device=device)