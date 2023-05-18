from A3C_network import Actor, Critic
import numpy as np
import gymnasium as gym
import torch.multiprocessing as mp
import torch

Max_episodes = 1000
buffer_size = 5
Max_actions = 300
gamma = 0.99


class worker(mp.Process):
    def __init__(self,globalActor,globalCritic,optimizerActor,optimizerCritic,globalEpCount,queue_avg,queue_ep,queue_lossActor,queue_lossCritic,globalRCount,name,environment='Ant-v4'):
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
        
        self.globalActor = globalActor
        self.globalCritic = globalCritic
        self.globalEpCount = globalEpCount
        self.queue_avg = queue_avg
        self.queue_ep = queue_ep
        self.queue_lossActor = queue_lossActor
        self.queue_lossCritic = queue_lossCritic
        self.globalRCount = globalRCount
        self.name = 'w'+str(name)
        self.env = gym.make(environment).unwrapped
        self.localActor = Actor(globalActor.s_dim,globalActor.a_dim)
        self.localActor.load_state_dict(globalActor.state_dict())
        self.localCritic = Critic(globalCritic.s_dim,globalCritic.a_dim)
        self.localCritic.load_state_dict(globalCritic.state_dict())
        self.optActor = optimizerActor
        self.optCritic = optimizerCritic


    def run(self):        
        while self.globalEpCount.value < Max_episodes:
            state = torch.tensor(self.env.reset()[0])

            buffer_a, buffer_s, buffer_r = [],[],[]
            ep_r = 0.
            done = False

            for t in range(Max_actions):

                action = self.localActor.select_action(state.float())
                new_state, reward, done, _, _ = self.env.step(action)
                #print(new_state==state,reward)
                new_state = torch.tensor(new_state)

                if t == Max_actions-1:
                    done = True
                
                buffer_a.append(action)
                buffer_r.append(reward)
                buffer_s.append(state)

                ep_r += reward
                state = new_state.clone()

                if done or (t%buffer_size == 0):

                    if done:
                        R = 0.
                    else:
                        R = self.localCritic.forward(new_state.float())
                        R = R.detach().numpy()[0]
                    
                    buffer_R = []

                    for r in buffer_r[::-1]:
                        R = r + gamma*R
                        buffer_R.append(R)

                    buffer_R.reverse()
                    
                    buffer_a = torch.tensor(np.array(buffer_a), dtype=torch.float32)
                    buffer_R = torch.tensor(buffer_R, dtype=torch.float32).reshape(-1,1)
                    buffer_s = torch.stack(buffer_s).float()

                    buffer_value = self.localCritic.forward(buffer_s)

                    lossActor = self.localActor.loss_func(buffer_s,buffer_a,buffer_R,buffer_value)
                    lossCritic = self.localCritic.loss_func(buffer_s,buffer_R)

                    
                    self.optActor.zero_grad()
                    self.optCritic.zero_grad()

                    lossActor.backward()
                    lossCritic.backward()

                    for loc, glob in zip(self.localActor.parameters(), self.globalActor.parameters()):
                          glob._grad = loc.grad.clone() 

                    for loc, glob in zip(self.localCritic.parameters(), self.globalCritic.parameters()):
                          glob._grad = loc.grad.clone()                        
                    
                    self.optActor.step()
                    self.optCritic.step()

                    self.localActor.load_state_dict(self.globalActor.state_dict())
                    self.localCritic.load_state_dict(self.globalCritic.state_dict())

                    buffer_a, buffer_s, buffer_r = [],[],[]

                    if done: #This is just for recording training progress
                        with self.globalEpCount.get_lock():
                            self.globalEpCount.value += 1
                        with self.globalRCount.get_lock():
                            if self.globalRCount.value == 0.:
                                self.globalRCount.value = ep_r
                            else:
                                self.globalRCount.value = self.globalRCount.value * 0.99 + ep_r * 0.01
                            self.queue_avg.put(self.globalRCount.value)
                            self.queue_ep.put(ep_r)
                            self.queue_lossActor.put(lossActor.detach().numpy())
                            self.queue_lossCritic.put(lossCritic.detach().numpy())

                        print(
                            self.name,
                            "Ep:", self.globalEpCount.value,
                            "| Ep_r: %.0f" % self.globalRCount.value,
                            f'| Last episode reward: {ep_r}'
                        )

                        break
        
        self.queue_avg.put(None)
        self.queue_ep.put(None)
        self.queue_lossActor.put(None)
        self.queue_lossCritic.put(None)