import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self,s_dim,a_dim):
        '''
        This class is meant to follow the network used for continous action control in the paper 
        "Asynchronous Methods for Deep Reinforcement Learning". It uses two networks to output a mean and a variance for each action
        in order to approximate the optimal policy and another network to approximate the value function.

        s_dim: The dimension of the state-vector, int-type
        a_dim: The dimension of the action-vector, int-type
        ''' 
        super(Actor, self).__init__()
        self.s_dim = s_dim 
        self.a_dim = a_dim  

        self.LinActor = nn.Linear(self.s_dim,256)
        torch.nn.init.normal_(self.LinActor.weight, mean=0.0, std=0.01)
        self.LinActor2 = nn.Linear(256,128)
        torch.nn.init.normal_(self.LinActor2.weight, mean=0.0, std=0.01)
        self.LinActor3 = nn.Linear(128,64)
        torch.nn.init.normal_(self.LinActor3.weight, mean=0.0, std=0.01)

        self.mu = nn.Linear(64,self.a_dim)
        torch.nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
        self.sigma = nn.Linear(64,self.a_dim)
        torch.nn.init.normal_(self.sigma.weight, mean=0.0, std=0.01)

        self.distribution = Normal
    
    def forward(self,x):
        a1 = nn.ReLU()(self.LinActor(x))
        a1 = nn.ReLU()(self.LinActor2(a1))
        a1 = nn.ReLU()(self.LinActor3(a1))      

        mean = self.mu(a1)
        variance = nn.Softplus()(self.sigma(a1))+0.0001

        return mean, variance
    
    def select_action(self,x):
        self.eval()

        mean, variance = self.forward(x)
        action = self.distribution(mean,variance).rsample()
        
        return action.detach().numpy()

    def loss_func(self,s,a,R,value):
        self.train()

        mean, variance = self.forward(s)
        advantage = R-value
        
        dist = self.distribution(mean,variance)
        log_prob = dist.log_prob(a)
        #print(f'log prob: {log_prob} \n mean: {mean} \n variance: {variance} \n action: {a}')
        entropy = -0.5*(torch.log(2*np.pi*variance) + 1)
        lossActor = (log_prob*advantage.detach() + entropy*1e-4)

        return lossActor.mean()

class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        '''
        This class is meant to follow the network used for continous action control in the paper 
        "Asynchronous Methods for Deep Reinforcement Learning". It uses two networks to output a mean and a variance for each action
        in order to approximate the optimal policy and another network to approximate the value function.

        s_dim: The dimension of the state-vector, int-type
        a_dim: The dimension of the action-vector, int-type
        ''' 
        super(Critic, self).__init__()
        self.s_dim = s_dim 
        self.a_dim = a_dim  

        self.LinCritic = nn.Linear(self.s_dim,256)
        torch.nn.init.normal_(self.LinCritic.weight, mean=0.0, std=0.01)
        self.LinCritic2 = nn.Linear(256,128)
        torch.nn.init.normal_(self.LinCritic2.weight, mean=0.0, std=0.01)
        self.LinCritic3 = nn.Linear(128,64)
        torch.nn.init.normal_(self.LinCritic3.weight, mean=0.0, std=0.01)
        
        self.value = nn.Linear(64,1)
        torch.nn.init.normal_(self.value.weight, mean=0.0, std=0.01)

        self.distribution = Normal
    
    def forward(self,x):
        c1 = nn.ReLU()(self.LinCritic(x))
        c1 = nn.ReLU()(self.LinCritic2(c1))
        c1 = nn.ReLU()(self.LinCritic3(c1))
        
        value = self.value(c1)
        return value

    def loss_func(self,s,R):
        self.train()

        value = self.forward(s)

        advantage = R-value
        lossCritic = advantage.pow(2)

        return lossCritic.mean()
    

if __name__ == '__main__':

    s = torch.randn(3, 5)  # shape: (batch_size, s_dim)
    a = torch.randn(3, 3)  # shape: (batch_size, a_dim)
    R = torch.randn(3, 1)  # shape: (batch_size, 1)

 
    act = Actor(5,3)
    crit = Critic(5,3)

    value = crit.forward(s)
    actloss = act.loss_func(s,a,R,value)
    critloss = crit.loss_func(s,R)

    #print(actloss,critloss)