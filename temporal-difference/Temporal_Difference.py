
#!/usr/bin/env python
# coding: utf-8

# # Temporal-Difference Methods
#
# In this notebook, you will write your own implementations of many Temporal-Difference (TD) methods.
#
# While we have provided some starter code, you are welcome to erase these hints and write your code from scratch.
#
# ---
#
# ### Part 0: Explore CliffWalkingEnv
#
# We begin by importing the necessary packages.

# In[ ]:


import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import check_test
from plot_utils import plot_values
import ipdb
from IPython import embed

# Use the code cell below to create an instance of the [CliffWalking](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py) environment.

# In[ ]:


env = gym.make('CliffWalking-v0')


# The agent moves through a $4\times 12$ gridworld, with states numbered as follows:
# ```
# [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#  [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
#  [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
#  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]
# ```
# At the start of any episode, state `36` is the initial state.  State `47` is the only terminal state, and the cliff corresponds to states `37` through `46`.
#
# The agent has 4 potential actions:
# ```
# UP = 0
# RIGHT = 1
# DOWN = 2
# LEFT = 3
# ```
#
# Thus, $\mathcal{S}^+=\{0, 1, \ldots, 47\}$, and $\mathcal{A} =\{0, 1, 2, 3\}$.  Verify this by running the code cell below.

# In[ ]:
#plt.ion()

print(env.action_space)
print(env.observation_space)
#ipdb.set_trace()

# In this mini-project, we will build towards finding the optimal policy for the CliffWalking environment.  The optimal state-value function is visualized below.  Please take the time now to make sure that you understand _why_ this is the optimal state-value function.
#
# _**Note**: You can safely ignore the values of the cliff "states" as these are not true states from which the agent can make decisions.  For the cliff "states", the state-value function is not well-defined._

# In[ ]:


# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

#plot_values(V_opt)


# ### Part 1: TD Control: Sarsa
#
# In this section, you will write your own implementation of the Sarsa control algorithm.
#
# Your algorithm has four arguments:
# - `env`: This is an instance of an OpenAI Gym environment.
# - `num_episodes`: This is the number of episodes that are generated through agent-environment interaction.
# - `alpha`: This is the step-size parameter for the update step.
# - `gamma`: This is the discount rate.  It must be a value between 0 and 1, inclusive (default value: `1`).
#
# The algorithm returns as output:
# - `Q`: This is a dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated action value corresponding to state `s` and action `a`.
#
# Please complete the function in the code cell below.
#
# (_Feel free to define additional functions to help you to organize your code._)

# In[ ]:


def sample_epsilon_greedy_policy(Q, state, epsilon, nA):
    """ samples from epsilon greedy policy """
    # how to get the total number of actions
    rand_val = np.random.random()

    if rand_val < epsilon:
        # pick a random value from the actions
        action = np.random.randint(0,nA)
    else:
        action = np.argmax(Q[state])
        value = Q[state][action]
    return action

def Q_update(Qsa, reward, Qnew_sa, alpha, gamma ):
    """
    Will it update the dictionary
    """
    new_Qsa = Qsa + alpha * (reward + gamma * Qnew_sa - Qsa)
    return new_Qsa


def sarsa(env, num_episodes, alpha, gamma=1.0, num_step_per_episode=300):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.action_space.n
    state = env.reset()
    epsilon = 0.1


    # initialize performance monitor
    plot_every = 100
    n_steps = 0
    tmp_scores = np.zeros((100,1))

    # monitor performance
    tmp_scores = deque(maxlen = plot_every)
    avg_scores = deque(maxlen = num_episodes)

    # loop over episodes
    for i_episode in range(1, num_episodes+1):

        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            epsilon = min(0.5, epsilon + .1)

        score = 0 # keeps score for each episode.
        state = env.reset()
        eps = 1.0 / i_episode

        action = sample_epsilon_greedy_policy(Q, state, epsilon, nA)

        while True:
            next_state, reward, done, info = env.step(action)
            score +=reward

            if done:
                Qnew_sa = 0
                Qsa = Q[state][action]
                Q[state][action] = Q_update(Qsa, reward, Qnew_sa, alpha, gamma)
                tmp_scores.append(score) # this is the total score in the episode
                break
            else:
                Qsa = Q[state][action]
                new_action = sample_epsilon_greedy_policy(Q, next_state, epsilon, nA)
                Qnew_sa = Q[next_state][new_action]
                Q[state][action] = Q_update(Qsa, reward, Qnew_sa, alpha, gamma)
                state = next_state
                action = new_action
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


# Use the next code cell to visualize the **_estimated_** optimal policy and the corresponding state-value function.
#
# If the code cell returns **PASSED**, then you have implemented the function correctly!  Feel free to change the `num_episodes` and `alpha` parameters that are supplied to the function.  However, if you'd like to ensure the accuracy of the unit test, please do not change the value of `gamma` from the default.

# In[ ]:


if 0:
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 500, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


#ipdb.set_trace()

# ### Part 2: TD Control: Q-learning
#
# In this section, you will write your own implementation of the Q-learning control algorithm.
#
# Your algorithm has four arguments:
# - `env`: This is an instance of an OpenAI Gym environment.
# - `num_episodes`: This is the number of episodes that are generated through agent-environment interaction.
# - `alpha`: This is the step-size parameter for the update step.
# - `gamma`: This is the discount rate.  It must be a value between 0 and 1, inclusive (default value: `1`).
#
# The algorithm returns as output:
# - `Q`: This is a dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated action value corresponding to state `s` and action `a`.
#
# Please complete the function in the code cell below.
#
# (_Feel free to define additional functions to help you to organize your code._)

# In[ ]:

def update_Q_sarsamax(Q, state, action, reward, next_state, alpha, gamma):

    current = Q[state][action]
    if next_state is not None:
        max_action = np.argmax(Q[next_state])
        Qsa_next =  Q[next_state][max_action]
    else:
        Qsa_next = 0

    new_value = current + alpha *(reward + gamma * Qsa_next - current)
    return new_value

def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.action_space.n
    # plot average score
    avg_scores = deque(maxlen=num_episodes)
    tmp_scores = deque(maxlen=plot_every)
    # does it reset..
    # loop over episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % plot_every == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            avg_scores.append(np.mean(tmp_scores))

        eps = 1.0/i_episode
        scores = 0
        state = env.reset()

        while True:
            # choose action from epsilon-greedy
            action = sample_epsilon_greedy_policy(Q, state, eps, nA)
            # Take action and observe R,S
            next_state, reward, done, info = env.step(action)
            scores += reward
            # update Q
            Q[state][action] = update_Q_sarsamax(Q, state, action, reward, next_state, alpha, gamma)
            state = next_state
            if done:
                # maintain reward for each episode
                tmp_scores.append(scores)
                break

    #plot the results you are plotting the reward over episodes
    x_axis = np.linspace(0,num_episodes, len(avg_scores), endpoint=False)
    plt.plot(x_axis, np.asarray(avg_scores))
    plt.xlabel("Episode number")
    plt.ylabel("Average Rewards (Over Next %d Episodes)" % plot_every)
    plt.show()

    #
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))
    return Q


# Use the next code cell to visualize the **_estimated_** optimal policy and the corresponding state-value function.
#
# If the code cell returns **PASSED**, then you have implemented the function correctly!  Feel free to change the `num_episodes` and `alpha` parameters that are supplied to the function.  However, if you'd like to ensure the accuracy of the unit test, please do not change the value of `gamma` from the default.

# In[ ]:

if 0:
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

#ipdb.set_trace()

# ### Part 3: TD Control: Expected Sarsa
#
# In this section, you will write your own implementation of the Expected Sarsa control algorithm.
#
# Your algorithm has four arguments:
# - `env`: This is an instance of an OpenAI Gym environment.
# - `num_episodes`: This is the number of episodes that are generated through agent-environment interaction.
# - `alpha`: This is the step-size parameter for the update step.
# - `gamma`: This is the discount rate.  It must be a value between 0 and 1, inclusive (default value: `1`).
#
# The algorithm returns as output:
# - `Q`: This is a dictionary (of one-dimensional arrays) where `Q[s][a]` is the estimated action value corresponding to state `s` and action `a`.
#
# Please complete the function in the code cell below.
#
# (_Feel free to define additional functions to help you to organize your code._)

# In[ ]:

def get_expected_q(Q, state, nA, eps):
    policy_s = np.ones(nA)*eps/nA
    max_action = np.argmax(Q[state])
    policy_s[max_action]= (1-eps) + eps/nA
    expected_q = np.dot(policy_s,Q[state])
    return expected_q


def update_Q_expected_sarsa(Q, state, action, reward, next_state, alpha, gamma, nA, eps):

    current = Q[state][action]
    if next_state is not None:
        Qsa_next = get_expected_q(Q, next_state, nA, eps)
    else:
        Qsa_next = 0

    new_value = current + alpha *(reward + gamma * Qsa_next - current)
    return new_value

def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    nA = env.action_space.n
    # plot average score
    avg_scores = deque(maxlen=num_episodes)
    tmp_scores = deque(maxlen=plot_every)
    # does it reset..
    # loop over episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % plot_every == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
            avg_scores.append(np.mean(tmp_scores))

        eps = 0.005 #1.0/i_episode
        scores = 0
        state = env.reset()

        while True:
            # choose action from epsilon-greedy
            action = sample_epsilon_greedy_policy(Q, state, eps, nA)
            # Take action and observe R,S
            next_state, reward, done, info = env.step(action)
            scores += reward
            # update Q
            Q[state][action] = update_Q_expected_sarsa(Q, state, action, reward, next_state, alpha, gamma, nA, eps)
            state = next_state
            if done:
                # maintain reward for each episode
                tmp_scores.append(scores)
                break

    #plot the results you are plotting the reward over episodes
    x_axis = np.linspace(0,num_episodes, len(avg_scores), endpoint=False)
    plt.plot(x_axis, np.asarray(avg_scores))
    plt.xlabel("Episode number")
    plt.ylabel("Average Rewards (Over Next %d Episodes)" % plot_every)
    plt.show()
    return Q

def expected_sarsa2(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))

    tmp_scores = deque(maxlen=plot_every)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function

    return Q


# Use the next code cell to visualize the **_estimated_** optimal policy and the corresponding state-value function.
#
# If the code cell returns **PASSED**, then you have implemented the function correctly!  Feel free to change the `num_episodes` and `alpha` parameters that are supplied to the function.  However, if you'd like to ensure the accuracy of the unit test, please do not change the value of `gamma` from the default.

# In[ ]:


# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 10000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
