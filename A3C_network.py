import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class Network(nn.Module):
    def __init__(self,s_dim,a_dim):
        '''
        This class is meant to follow the network used for continous action control in the paper 
        "Asynchronous Methods for Deep Reinforcement Learning". It uses two networks to output a mean and a variance for each action
        in order to approximate the optimal policy and another network to approximate the value function.

        s_dim: The dimension of the state-vector, int-type
        a_dim: The dimension of the action-vector, int-type
        ''' 
        super(Network, self).__init__()
        self.s_dim = s_dim 
        self.a_dim = a_dim  

        self.LinActor = nn.Linear(self.s_dim,200) 
        self.mu = nn.Linear(200,self.a_dim)
        self.sigma = nn.Linear(200,self.a_dim)

        self.LinCritic = nn.Linear(self.s_dim,200)
        self.value = nn.Linear(200,1)

        self.distribution = MultivariateNormal
    
    def forward(self,x):
        a1 = nn.ReLU6()(self.LinActor(x))
        mean = 2*nn.Tanh()(self.mu(a1))
        covariance = nn.Softplus()(self.sigma(a1))
        covariance = torch.diag_embed(covariance) + 0.001

        c1 = nn.ReLU6()(self.LinCritic(x))
        value = self.value(c1)

        return mean, covariance, value
    
    def select_action(self,x):
        self.eval()

        mean, covariance, _ = self.forward(x)
        action = self.distribution(mean,covariance).sample()

        return action.numpy()

    def loss_func(self,s,a,R):
        self.train()

        mean, covariance, value = self.forward(s)
        advantage = R-value
        lossCritic = advantage.pow(2)

        dist = self.distribution(mean,covariance)
        log_prob = dist.log_prob(a).reshape(-1,1)

        covar_diag = torch.diagonal(covariance,offset=0,dim1=1,dim2=2)
        entropy = -0.5*(torch.log(2*np.pi*covar_diag) + 1)
        lossActor = (log_prob*advantage.detach() + entropy*1e-4)

        return (lossActor + lossCritic).mean()
    
if __name__ == '__main__':
    net = Network(5,3)

    s = torch.randn(3, 5)  # shape: (batch_size, s_dim)
    a = torch.randn(3, 3)  # shape: (batch_size, a_dim)
    R = torch.randn(3, 1)  # shape: (batch_size, 1)

    # Call the loss_func method and check the output shape
    loss = net.loss_func(s, a, R)
    print(loss)