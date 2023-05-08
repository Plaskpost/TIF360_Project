from A3C_network import Network
import numpy as np
import gymnasium as gym
import torch.multiprocessing as mp
import torch


Max_episodes = 1000
batch_size = 50
Max_actions = 300
gamma = 0.9

class worker(mp.Process):
    def __init__(self,globalNetwork,optimizer,globalEpCount,queue,globalRCount,name,environment='Ant-v4'):
        '''
        This class implements the worker for a A3C reinforcement learning algorithm.

        globalNetwork: The global network, type: suitable pytorch network
        self.globalEpCount: Global episode counter,type:  multiprocessing.Value
        queue: multiprocessing queue meant to send results between processess, type: multiprocessing.queue
        globalRCount: keeps track of global reward,type:  multiprocessing.Value
        name: just the name of the worker, type: string
        environment: which mujoco environment we wish to run, type: suitable mujoco gym enviroment
        '''
        super(worker, self).__init__()
        
        self.globalNetwork = globalNetwork
        self.globalEpCount = globalEpCount
        self.queue = queue
        self.globalRCount = globalRCount
        self.name = 'w'+str(name)
        self.env = gym.make(environment).unwrapped #, render_mode='human'
        self.localNetwork = Network(globalNetwork.s_dim,globalNetwork.a_dim)
        self.opt = optimizer


    def run(self):        
        while self.globalEpCount.value < Max_episodes:
            self.opt.zero_grad()
            state = torch.tensor(self.env.reset()[0])
            self.localNetwork.load_state_dict(self.globalNetwork.state_dict())

            buffer_a, buffer_s, buffer_r = [],[],[]
            ep_r = 0
            done = False

            for t in range(Max_actions):

                action = self.localNetwork.select_action(state.float())
                new_state, reward, done, _, _ = self.env.step(action)

                new_state = torch.tensor(new_state)

                if t == Max_actions-1:
                    done = True
                
                buffer_a.append(action)
                buffer_r.append((reward+8.1)/8.1)
                buffer_s.append(state)

                ep_r += reward

                if done or (t%batch_size == 0):

                    if done:
                        R = 0.
                    else:
                        _,_,R = self.localNetwork.forward(new_state.float())
                        R = R.detach().numpy()[0]
                    
                    buffer_R = []

                    for r in buffer_r[::-1]:
                        R = r + gamma*R
                        buffer_R.append(R)
                    
                    buffer_a = torch.tensor(np.array(buffer_a), dtype=torch.float32)
                    buffer_R = torch.tensor(buffer_R, dtype=torch.float32).reshape(-1,1)
                    buffer_s = torch.stack(buffer_s).float()
                  
                    loss = self.localNetwork.loss_func(buffer_s,buffer_a,buffer_R) 

                    self.opt.zero_grad()
                    loss.backward()

                    for loc, glob in zip(self.localNetwork.parameters(), self.globalNetwork.parameters()):
                          glob._grad = loc.grad.clone()
                    
                    self.opt.step()

                    self.localNetwork.load_state_dict(self.globalNetwork.state_dict())

                    buffer_a, buffer_s, buffer_r = [],[],[]

                    if done: #This is just for recording training progress
                        with self.globalEpCount.get_lock():
                            self.globalEpCount.value += 1
                        with self.globalRCount.get_lock():
                            if self.globalRCount.value == 0.:
                                self.globalRCount.value = ep_r
                            else:
                                self.globalRCount.value = self.globalRCount.value * 0.99 + ep_r * 0.01
                            self.queue.put(self.globalRCount.value)
                        print(
                            self.name,
                            "Ep:", self.globalEpCount.value,
                            "| Ep_r: %.0f" % self.globalRCount.value,
                        )

                        break
        self.queue.put(None)