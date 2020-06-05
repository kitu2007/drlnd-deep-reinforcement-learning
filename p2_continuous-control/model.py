import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

def bn_func(fn, batch_norm=False):
    if batch_norm:
        return fn
    else:
        return lambda x:x


class Critic(nn.Module):
    """given a state and action it will give a value"""
    def __init__(self, state_size, action_size, seed, fc1=400, fc2=300, use_batch_norm = False, bn_after_act=False):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        self.bn_after_act = bn_after_act
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1+action_size, fc2)
        self.fc3 = nn.Linear(fc2,1)
        if self.use_batch_norm:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(fc1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)

    def forward(self, state, action):
        if self.use_batch_norm:
            if self.bn_after_act:
                  xs = self.bn1(F.relu(self.fc1(self.bn0(state))))
                  x = torch.cat((xs, action), dim=1)
                  x = F.relu(self.fc2(x))
            else:
                # this is the normal way to apply batch norm (before activation)
                xs = F.relu(self.bn1(self.fc1(self.bn0(state))))
                x = torch.cat((xs, action), dim=1)
                x = F.relu(self.fc2(x))
        else:
            xs = F.relu(self.fc1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Given a state it will propose an action"""

    def __init__(self, state_size, action_size, seed, fc1=400, fc2=300, use_batch_norm=False, bn_after_act=False):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        self.bn_after_act = bn_after_act
        self.fc1 = nn.Linear(state_size,fc1)
        self.fc2 = nn.Linear(fc1,fc2)
        self.fc3 = nn.Linear(fc2,action_size)
        if self.use_batch_norm:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(fc1)
            self.bn2 = nn.BatchNorm1d(fc2)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if self.use_batch_norm:
            if self.bn_after_act:
                x = self.bn1(F.relu(self.fc1(self.bn0(state))))
                x = self.bn2(F.relu(self.fc2(x)))
            else:
                # this is the normal way to apply batch norm (before activation)
                x = F.relu(self.bn1(self.fc1(self.bn0(state))))
                x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
