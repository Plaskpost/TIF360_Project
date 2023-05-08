import torch.optim as optim
from torch.multiprocessing import Value, Queue
import torch.multiprocessing as mp
from A3C_network import Network
import A3C_worker
import gymnasium as gym
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    env_name = 'Ant-v4'
    n_workers = mp.cpu_count()

    env = gym.make(env_name, render_mode='human')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    global_network = Network(s_dim, a_dim)
    global_network.share_memory() 

    optimizer = optim.Adam(global_network.parameters(), lr=0.0001)

    global_ep_count = Value('i', 0)
    global_reward_count = Value('d', 0.0) 
    queue = Queue() 

    workers = [A3C_worker.worker(global_network, optimizer, global_ep_count, queue, global_reward_count, i, env_name) for i in range(n_workers)] # create worker processes

    [w.start() for w in workers]

    res = []

    while True:
        r = queue.get() 
        if r is not None:
            res.append(r)
        else:
            break

    plt.plot(res)
    plt.show()    

    [w.join() for w in workers]    
    
    state = torch.tensor(env.reset()[0])
    done = False
    while not done:
        env.render()
        action = global_network.select_action(state.float())
        state, reward, done, _, _ = env.step(action)
        state = torch.tensor(state)