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

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4 # Learning rate of actor
LR_CRITIC = 1e-3 # Learning rate of the critic
WEIGHT_DECAY = 0.001 # L2 weight decay (regularization)
clip_grad_value = 1.0
LEARN_AFTER_N_STEPS = 20
NUM_LEARN_STEPS = 10

print_var_list = ['BUFFER_SIZE', 'BATCH_SIZE', 'TAU', 'LR_ACTOR', 'LR_CRITIC', 'WEIGHT_DECAY', 'clip_grad_value', 'LEARN_AFTER_N_STEPS', 'NUM_LEARN_STEPS']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_current_date_time(just_get_day=False):
    # datetime object containing current date and time
    now = datetime.now()
    if just_get_day:
        dt_string = now.strftime("%d_%m_%Y")
    else:
        dt_string = now.strftime("%d_%m_%Y_%H:%M")
    return dt_string

class json_logger:

    def __init__(self,dirname = 'OutFiles', filename='json_logger.json'):
        self.fname =   dirname + '/' + get_current_date_time() + '_' + filename
        json_object = OrderedDict()
        self.add(json_object)


    def add(self, values):
        json_object = values
        with open(self.fname, "w") as outfile:
            json.dump(json_object, outfile, indent = 4, ensure_ascii = False)

    def update(self, values):
        with open(self.fname, 'r') as openfile:
            json_object = json.load(openfile)

        json_object.update(values)
        with open(self.fname, 'w') as outfile:
            json.dump(json_object, outfile, indent = 4, ensure_ascii = False)


def init_logger(dirname = 'OutFiles', filename='logger.txt', level=logging.DEBUG):
    fname =  dirname + '/' + get_current_date_time() + '_' + filename
    return fname



def write_config_files_json(dirname = 'OutFiles', filename='json_config.txt', var_list=print_var_list):
    variables = globals()
    jsonData = OrderedDict()
    for var in var_list:
        #ipdb.set_trace()
        if var in print_var_list and var in variables:
            jsonData[var] = variables[var]

    filename =  dirname + '/' + get_current_date_time() + '_' + filename
    with open(filename, 'w') as outfile:
        json.dump(jsonData, outfile, sort_keys = True, indent = 4,
                  ensure_ascii = False)

    return jsonData

def write_config_files(dirname = 'OutFiles', filename='config.txt', var_list=print_var_list):
    variables = globals()
    filename =  dirname + '/' + get_current_date_time() + '_' + filename
    outfile = open(filename, 'w')
    outfile.write("********** START CONFIG VALUES***********")
    for var in var_list:
        if var in print_var_list and var in variables:
            outfile.write('{}={}\n'.format(var, str(variables[var])))

    outfile.write("**********END CONFIG VALUES***********")
    outfile.close()
    return


def normalize_data(x,axis=0, eps=1e-5):
    mu = x.mean()
    sigma = x.std() + eps
    x = (x - mu[:,np.newaxis])/sigma[:,np.newaxis]
    return x, mu, sigma

class Agent():
    def __init__(self, state_size, action_size, random_seed, num_agents, use_batch_norm):
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


        self.actor_target = Actor(self.state_size, self.action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.actor_local  = Actor(self.state_size, self.action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_target = Critic(self.state_size, self.action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.critic_local  = Critic(self.state_size, self.action_size, random_seed, use_batch_norm=use_batch_norm).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        self.memory_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise = ULNoise(self.action_size, random_seed)
        self.nsteps = 0
        self.num_agents = num_agents

    def step(self, state, action, reward, next_state, done):
         #insert into the memory buffer
        self.memory_buffer.add(state, action, reward, next_state, done)
        self.nsteps +=1
        if len(self.memory_buffer) > BATCH_SIZE:
            experiences = self.memory_buffer.sample()
            if self.nsteps % LEARN_AFTER_N_STEPS == 0:
                for i in range(NUM_LEARN_STEPS):
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
        Q_targets = Q_targets.detach()
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),clip_grad_value)
        self.critic_optimizer.step()

        # ----- train the actor method
        # update the actor
        actions_pred = self.actor_local(states)
        # mean of the action_pred of that should be zero
        actor_loss = -self.critic_local(states.detach(), actions_pred).mean()
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
        dx = self.theta *(self.mu - x) + self.sigma * np.random.rand(len(x))
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
        for i in range(state.shape[0]):
            e = self.experience(state[i], action[i], reward[i], new_state[i], done[i])
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
