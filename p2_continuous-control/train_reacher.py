



import random
import torch
import numpy as np
from collections import deque
from IPython import embed


#!/usr/bin/env python
import ipdb


from ddpg_agent import Agent

from unityagents import UnityEnvironment
import numpy as np

# Instantiate the environment
env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

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



def ddpg(agent, n_episodes=20, max_t=600):
    """
    DDPG Agent

    """
    scores = list()#np.zeros(num_agents)
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions  = agent.act(states)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            score += env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done
            #ipdb.set_trace()
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break

        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(i_episode, np.mean(scores_window)))
            #print(scores_window)
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint_1.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint_1.pth')
            break
    return scores


agent = Agent(state_size, action_size, random_seed=2, num_agents=20, use_batch_norm=True)


if train_agent:
    scores = ddpg(agent, n_episodes=10000)
    ipdb.set_trace()
    #plot the scores.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ipdb.set_trace()
    plt.plot(np.arange(len(scores)),np.mean(scores))
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
