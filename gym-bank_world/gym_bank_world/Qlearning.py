from csv import writer
import numpy as np
import gym
from matplotlib import pyplot as plt
import pickle
import logging
import os
import random
import cv2
import pickle as pkl

# Set learning parameters
learning_rate = .01
discount_factor = .99
num_test_episodes = 100
log_step = 10000
total_steps_per_task = 1500000  # 2e6
test_epsilon = 0.01



def train(train_env,action_space= 5, num_episodes = 1000, q_path = None):
    # create lists to contain total rewards and steps per episode
    episode_rewards = []
    random_ep = np.arange(0.01, 0.75, 0.99 / num_episodes, dtype=float)[::-1]
    steps = 0
    i = 0

    Q_table = {}
    start_episode = 0
    if q_path is not None:
        start_episode = int(q_path.split('_')[-1].split('.')[0])
        Q_table = pkl.load(open(q_path, "rb"))
    for episode in range(start_episode, num_episodes):
        # Reset environment
        train_env.reset()
        # train_env.render()
        ep = random_ep[min(i, random_ep.shape[0] - 1)]
        i += 1

        # Get state representation
        s = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)
        while not train_env.done:

            # Get action
            a_list = get_action(Q_table, action_space, ep, s, train_env.no_of_agents)

            # take step in env and get reward
            train_env.step(a_list)
            # train_env.render()

            # Get new state
            s_prime = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)


            # update Q_table
            update_qvalue(Q_table, action_space, s, a_list, train_env.total_reward,  s_prime, train_env.no_of_agents)

            s = s_prime

        episode_rewards.append(train_env.total_reward)
        print(f"Episode {episode + 1} reward: ", train_env.total_reward)
        plt.clf()
        plt.plot(episode_rewards)
        plt.savefig("D:\RL_project\Episode_reward_plot.png")
        if (episode+1)%10 == 0:
            with open(f"D:\RL_project\QTable_{episode+1}.pkl", "wb") as fi:
                pkl.dump(Q_table, fi)

    cv2.destroyAllWindows()
    return Q_table, episode_rewards


def update_qvalue(Q_table, action_space, s, a_list, r, s_prime, no_of_agents):
    if s_prime in Q_table.keys():
        q_next = Q_table[s_prime]
    else:
        q_next = np.zeros((no_of_agents, action_space))
        Q_table[s_prime] = q_next
    if s not in Q_table.keys():
        Q_table[s] = np.zeros((no_of_agents, action_space))

    for i, a in enumerate(a_list):
        Q_table[s][i][a] = Q_table[s][i][a] + learning_rate * (r + discount_factor * np.max(q_next[i]) - Q_table[s][i][a])


def get_action(Q_table, action_space, ep, s, no_of_agents):
    if np.random.randn(0, 1) < ep:
        action_list = [random.randint(0, action_space - 1) for i in range(no_of_agents)]

    else:
        if s in Q_table:
            action_list = []
            for i in range(no_of_agents):
                where_max = np.where(Q_table[s][i] == np.max(Q_table[s][i]))[0]
                if len(where_max) == 1:
                    a = where_max[0]
                else:
                    a = np.random.choice(where_max)
                action_list.append(a)
        else:
            action_list = [random.randint(0, action_space - 1) for i in range(no_of_agents)]
    return action_list

def get_state_representation(s, no_of_agents, no_of_gems):
    rep = ""
    for i in range(no_of_agents):
        rep += np.array2string(s[f"A_{i}"][0]) + "," + str(s[f"A_{i}"][1]) + "," + str(s[f"A_{i}"][2]) + ";"
    rep += "_"
    for j in range(no_of_gems):
        rep += np.array2string(s[f"Gem_{j}"][0]) + "," + str(s[f"Gem_{j}"][1]) + "," + str(s[f"Gem_{j}"][2]) + ";"

    return rep


# def test(Q_table, test_env, display=False, action_space=4):
#     is_success = np.zeros(num_test_episodes)
#     episode_reward = np.zeros(num_test_episodes)
#     for i in range(num_test_episodes):
#         # Reset environment and get first new observation
#         s = test_env.reset()
#         total_reward = 0
#         plan_seq = test_env.plan_seq
#         for operator in plan_seq:
#             operator_index = test_env.operator_order.index(operator[0])
#             subgoal_done = test_env.is_terminal_state(operator, s)
#             s_hat = test_env.get_state_representation(s, operator)
#             Q_table = operator_Qtables[operator_index]
#             if display:
#                 test_env.render()
#             while not subgoal_done:
#
#                 # get action
#                 a = get_action(Q_table,action_space,test_epsilon,s_hat)
#
#                 # take step in env
#                 s_prime, r, done, info = test_env.step(a)
#
#                 # get abstract representation
#                 s_prime_hat = test_env.get_state_representation(s_prime, operator)
#
#                 # Check if the next state is terminal
#                 subgoal_done = test_env.is_terminal_state(operator, s_prime)
#                 if done:
#                     subgoal_done = done
#
#                 if display:
#                     test_env.render()
#                 s = s_prime
#                 s_hat = s_prime_hat
#                 total_reward += r
#         is_success[i] = info['is_success']
#         episode_reward[i] = total_reward
#     return np.average(is_success), np.average(episode_reward)
#
#
#
#
#
#


