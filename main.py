from torch.multiprocessing import Value, Queue
import torch.multiprocessing as mp
from A3C_network import Actor, Critic
from A3C_worker import worker
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch.optim import RMSprop

if __name__ == '__main__':
    env_name = "Ant-v4"

    n_workers = mp.cpu_count()

    env = gym.make(env_name, render_mode='human')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    globalActor = Actor(s_dim, a_dim)
    globalActor.share_memory() 
    globalCritic = Critic(s_dim, a_dim)
    globalCritic.share_memory()
   
    optimizerActor = RMSprop(globalActor.parameters(), lr=1e-5, maximize=True)
    optimizerCritic = RMSprop(globalCritic.parameters(), lr=1e-5)

    global_ep_count = Value('i', 0)
    global_reward_count = Value('d', 0.0)
    queue_avg = Queue()
    queue_ep = Queue()
    queue_lossAct = Queue()
    queue_lossCrit = Queue()

    workers = [worker(globalActor,
                      globalCritic,
                      optimizerActor,
                      optimizerCritic,
                      global_ep_count,
                      queue_avg,
                      queue_ep,
                      queue_lossAct,
                      queue_lossCrit,
                      global_reward_count,
                      i,
                      env_name) for i in range(n_workers)]

    [w.start() for w in workers]

    res = []
    ep_res = []
    lossAct = []
    lossCrit = []

    while True:
        r = queue_avg.get() 
        r2 = queue_ep.get()
        la = queue_lossAct.get()
        lc = queue_lossCrit.get()
        if r is not None:
            res.append(r)
            ep_res.append(r2)
            lossAct.append(la)
            lossCrit.append(lc)
        else:
            break

    plt.plot(res, label='moving average')
    plt.plot(ep_res,alpha=0.5,label='epsiode reward')
    plt.legend()
    plt.show()    

    plt.plot(lossAct, label='Loss Actor')
    plt.legend()
    plt.show()

    
    plt.plot(lossCrit, label='Loss Critic')
    plt.legend()
    plt.show()

    [w.join() for w in workers]
    
    state = torch.tensor(env.reset()[0])
    done = False
    final_pol_reward = 0

    while not done:
        env.render()
        action = globalActor.select_action(state.float())
        state, reward, done, _, _ = env.step(action)
        state = torch.tensor(state)
        final_pol_reward += reward