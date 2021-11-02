import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]).cuda(), act()]  #add on .cuda()
    return nn.Sequential(*layers).cuda()  #add on .cuda()

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        a= torch.as_tensor([1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0,1.0], device=torch.device('cuda'))
        if len(obs.size())==1:
            if obs[24].tolist()== -1:
                a[0],a[4],a[8]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if obs[25].tolist()== -1:
                a[1],a[5],a[9]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if obs[26].tolist()== -1:
                a[2],a[6],a[10]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if obs[27].tolist()== -1:
                a[3],a[7],a[11]=torch.as_tensor([0,0,0], device=torch.device('cuda')) #obs[24:28]
        elif len(obs.size())==2: #how to fix one batch with same only?  o2 previous 
        
            if int(obs[0][24].tolist())== -1:
                a[0],a[4],a[8]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if int(obs[0][25].tolist())== -1:
                a[1],a[5],a[9]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if int(obs[0][26].tolist())== -1:
                a[2],a[6],a[10]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
            if int(obs[0][27].tolist())== -1:
                a[3],a[7],a[11]=torch.as_tensor([0,0,0], device=torch.device('cuda'))
        else:
            print('something wrong with obs')
        #TODO MASKING AT HERE !!! #training no need mask, testing just masking TODO here 
        return self.pi(obs)  * a

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        ##act_limit = action_space.high[0]
        act_limit = 1.0

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)# give temp action, then based on obs  to mask , self.pi = masked the MLP
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)#
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)#


    def act(self, obs):
        with torch.no_grad():
            # if torch.cuda.is_available():  #extra
            #     dev = "cuda:0"  #
            #     print('CUDA using for obs') #
            # else:   #
            #     dev = "cpu"  # 
            #     print('cpu used for obs') #
            # obs.to(torch.device(dev)) # extra
            # obs.cuda()
            return self.pi(obs) #self.pi(obs).numpy()
