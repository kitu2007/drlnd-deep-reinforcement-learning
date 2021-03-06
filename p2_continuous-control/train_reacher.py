import random
import torch
import numpy as np
from collections import deque
from IPython import embed
import logging


#!/usr/bin/env python
import ipdb

import utils
from ddpg_agent import Agent

from unityagents import UnityEnvironment
import numpy as np

# Instantiate the environmen
#env = UnityEnvironment(file_name='Reacher1.app')
env = UnityEnvironment(file_name='Reacher1.app')
#env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
#env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


watch_untrained_agent = False
train_agent = True

if watch_untrained_agent:
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    ipdb.set_trace()
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# write config file and training scores
log_fname = utils.logger_fname()
logging.basicConfig(filename=log_fname,level=logging.DEBUG)
json_log = utils.json_logger(filename='json_logger.json')


def ddpg(agent, n_episodes=20, max_t=1000):
    """
    DDPG Agent

    """
    scores = list()#np.zeros(num_agents)
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        agent.reset()
        for t in range(max_t):
            actions  = agent.act(states)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            score += env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break

        scores_window.append(score)
        scores.append(score)
        mean_score = np.mean(scores_window)

        print('\rEpisode {}\tAverage Score: {:.4f} Score: {:.4f}'.format(i_episode, mean_score, score.mean()), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f} Score: {:.4f}'.format(i_episode, mean_score, score.mean()))
            json_log.update({i_episode:mean_score})
            #logging.info('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, mean_score))

            #print(scores_window)
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint_1.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint_1.pth')
            break
    return scores


def ddpg_1(agent, n_episodes=20, max_t=1000):
    """
    DDPG Agent for one agent

    """
    scores = list() #np.zeros(num_agents)
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action  = agent.act(state[np.newaxis,:])
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        mean_score = np.mean(scores_window)

        print('\rEpisode {}\tAverage Score: {:.4f} Score: {:.4f}'.format(i_episode, mean_score, score), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f} Score: {:.4f}'.format(i_episode, mean_score, score))
            ipdb.set_trace()
            json_log.update({i_episode:mean_score})
            #logging.info('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, mean_score))

            #print(scores_window)
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint_1.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint_1.pth')
            break
    return scores



agent = Agent(state_size, action_size, random_seed=10, num_agents=num_agents)


if train_agent:
    scores = ddpg(agent, n_episodes=1500)
    ipdb.set_trace()
    #plot the scores.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ipdb.set_trace()
    plt.plot(np.arange(len(scores)),np.mean(scores))
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

env.close()
