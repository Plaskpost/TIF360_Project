import gymnasium as gym
import torch
import numpy as np

env = gym.make('Ant-v4', render_mode='human')
state = env.reset()[0]

state = torch.tensor(state)

for t in range(1000):
    env.render()
    action = np.random.rand(8)
    state, reward, done, _, _ = env.step(action)
