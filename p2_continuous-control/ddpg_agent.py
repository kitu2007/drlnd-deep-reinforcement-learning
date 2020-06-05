import numpy as np
import random
import copy
from collections import namedtuple, deque, OrderedDict

from  model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import logging
import json
#from ipdb import IPython
from datetime import datetime
import utils

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4 # Learning rate of actor
LR_CRITIC = 1e-4 # Learning rate of the critic
WEIGHT_DECAY = 0 # L2 weight decay (regularization)
clip_grad_value = 1.0
LEARN_AFTER_N_STEPS = 20
NUM_LEARN_STEPS = 10
NOISE_DECAY=0.001
UL_THETA = 0.15
UL_SIGMA = 0.1
ACTOR_FC_LAYERS = [128,128]
CRITIC_FC_LAYERS = [128,128]
BN_AFTER_ACTIVATION = True
USE_BATCH_NORM = True
BN_NORMALISE_STATE = False,
PRINT_GRADIENT = False


print_var_list = ['BUFFER_SIZE', 'BATCH_SIZE', 'TAU', 'LR_ACTOR', 'LR_CRITIC', 'WEIGHT_DECAY', 'clip_grad_value', 'LEARN_AFTER_N_STEPS', 'NUM_LEARN_STEPS', 'NOISE_DECAY', 'UL_THETA', 'UL_SIGMA', 'ACTOR_FC_LAYERS', 'BN_AFTER_ACTIVATION', 'BN_NORMALISE_STATE']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_data(x,axis=0, eps=1e-5):
    mu = x.mean()
    sigma = x.std() + eps
    x = (x - mu[:,np.newaxis])/sigma[:,np.newaxis]
    return x, mu, sigma

class Agent():
    def __init__(self, state_size, action_size, random_seed, num_agents):
        """
        Initialize an Agent Object

        Params
        --------
        state_size (int): dimension of each state
        action_size(int): dimension of action
        random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        actor_kwargs = {'fc1': ACTOR_FC_LAYERS[0], 'fc2': ACTOR_FC_LAYERS[1],
                        'use_batch_norm': USE_BATCH_NORM, 'bn_after_act':BN_AFTER_ACTIVATION,
                        'bn_normalize_state': BN_NORMALISE_STATE}
        critic_kwargs = {'fc1': CRITIC_FC_LAYERS[0], 'fc2': CRITIC_FC_LAYERS[1],
                         'use_batch_norm': USE_BATCH_NORM, 'bn_after_act': BN_AFTER_ACTIVATION,
                         'bn_normalize_state': BN_NORMALISE_STATE}
        #actor_kwargs = {fc_layers= ACTOR_FC_LAYERS, 'use_batch_norm': use_batch_norm}
        #critic_kwargs = {'fc_layers': CRITIC_FC_LAYERS, 'use_batch_norm': use_batch_norm}

        self.actor_target = Actor(self.state_size, self.action_size, random_seed,
                                  **actor_kwargs).to(device)
        self.actor_local  = Actor(self.state_size, self.action_size, random_seed, **actor_kwargs).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_target = Critic(self.state_size, self.action_size, random_seed, **critic_kwargs).to(device)
        self.critic_local  = Critic(self.state_size, self.action_size, random_seed, **critic_kwargs).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        self.memory_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise = ULNoise(self.action_size, random_seed, theta= UL_THETA, sigma=UL_SIGMA)
        self.nsteps = 0
        self.learn_steps = 0
        self.num_agents = num_agents
        # write config files
        variables = globals()
        utils.write_config_files(variables, print_var_list,filename='config.txt')
        json_config = utils.write_config_files_json(variables, print_var_list,filename='json_config.txt')


    def step(self, states, actions, rewards, next_states, dones):
        #insert into the memory buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory_buffer.add(state, action, reward, next_state, done)

        self.nsteps +=1
        if len(self.memory_buffer) > BATCH_SIZE:
            if self.num_agents == 1:
                experiences = self.memory_buffer.sample()
                self.learn_steps +=1
                self.learn(experiences, GAMMA)
            else:
                if self.nsteps % LEARN_AFTER_N_STEPS == 0:
                    self.learn_steps +=1
                    for i in range(NUM_LEARN_STEPS):
                        experiences = self.memory_buffer.sample()
                        self.learn(experiences, GAMMA)
                        
    def act(self, state, add_noise=True):
        """what is the policy based on which it acts. at every step it acts"""
        # agent takes an action based on policy
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # put the net to training mode
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        # so lets see how the learning works. First it works from experiences.
        # agent act based on local network and uses target network (domain shift)
        #ipdb.set_trace()
        states, actions, rewards, next_states, dones = experiences
        # rewards_norm = normalize_data(rewards)

        # ----- train the critic method
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_local(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        Q_expected = self.critic_local(states, actions)
        # Q_targets = Q_targets.detach()
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        #ipdb.set_trace()
        #ipdb.set_trace=lambda:None

        #print("before max_gradient:", self.critic_local.fc1.weight.grad.max(), self.critic_local.fc2.weight.grad.max(), self.critic_local.fc3.weight.grad.max())
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if PRINT_GRADIENT:
            print("before optimizer max_gradient:", self.critic_local.fc1.weight.grad.max(), self.critic_local.fc2.weight.grad.max(), self.critic_local.fc3.weight.grad.max())
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),clip_grad_value)
        self.critic_optimizer.step()
        if PRINT_GRADIENT:
            print("after optimizer  max_gradient:", self.critic_local.fc1.weight.grad.max(), self.critic_local.fc2.weight.grad.max(), self.critic_local.fc3.weight.grad.max())
        # ----- train the actor method
        actions_pred = self.actor_local(states)
        # mean of the action_pred of that should be zero
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimise loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),clip_grad_value)
        self.actor_optimizer.step()

        ## soft update. update target networks.
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Update the networks"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ULNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta *(self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class NormalNoise:
    def __init__(self, size, seed, mu=0.0, sigma = 0.2):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def sample(self):
        noise = self.mu + self.sigma * np.random.randn(len(x))
        return noise

class ReplayBuffer:
    def __init__(self, max_len, batch_size, seed):
        """Pass """
        self.max_len = max_len
        self.memory = deque(maxlen= self.max_len)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.batch_size = batch_size

    def add(self, state, action, reward, new_state, done):
        """insert an experience into the memory buffer"""
        e = self.experience(state, action, reward, new_state, done)
        self.memory.append(e)


    def sample(self):
        """randomly sample the buffer to get a set of experiences based on batch_size"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # convert from numpy to torch to float to device
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones  = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of the internal memory"""
        return len(self.memory)
