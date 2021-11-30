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

random.seed(21)
np.random.seed(21)
# Set learning parameters
learning_rate = .01
discount_factor = .99
# num_test_episodes = 100
# log_step = 10000
# total_steps_per_task = 1500000  # 2e6
# test_epsilon = 0.01


def train(train_env, action_space=5, num_episodes=1000, q_path=None, verbose=False):
    # create lists to contain total rewards and steps per episode
    episode_rewards = []
    # random_ep = np.arange(0.01, 0.9, 0.99 / (num_episodes * 10), dtype=float)[::-1]
    random_ep = np.arange(0.01, 0.9, 0.001 , dtype=float)[::-1]
    i = 0

    QPickup = {}
    QDrop = {}
    Q_table = [QPickup, QDrop]
    start_episode = 0
    if q_path is not None:
        start_episode = int(q_path.split('_')[-1].split('.')[0])
        Q_table = pkl.load(open(q_path, "rb"))
    for episode in range(start_episode, num_episodes):
        print (f'# Episode {episode+1}:')

        # Reset environment
        train_env.reset()
        if verbose:
            train_env.render()
        ep = random_ep[min(i, random_ep.shape[0] - 1)]
        # ep = 1
        i += 1

        # # Get state representation
        # s = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)
        s = train_env.obs_dict
        while not train_env.done:
            # Get action
            a_list = get_action(Q_table, action_space, ep, s, train_env.no_of_agents)

            # take step in env and get reward
            train_env.step(a_list)
            if verbose:
                train_env.render()

            # # Get new state
            # s_prime = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)
            s_prime = train_env.obs_dict

            # update Q_table
            update_qvalue(Q_table, action_space, s, a_list, train_env.reward_list, s_prime, train_env.no_of_agents)

            for agent_idx in range(train_env.no_of_agents):
                I = agent_idx
                if train_env.picked[I]:
                    j = s[f"A_{I}"][1]
                    print (f'Agent {I} picked Gem {j}. Reward:', train_env.reward_list[I])
                if train_env.dropped[I]:
                    print(f'Agent {I} dropped at bank. Reward:', train_env.reward_list[I])

            s = s_prime


        # episode_rewards.append(train_env.total_reward)
        # print(f"Episode {episode + 1} reward: ", train_env.total_reward)
        # plt.clf()
        # plt.plot(episode_rewards)
        # plt.savefig("plot.png")
        if (episode + 1) % 10 == 0:
            with open(f"QTable_{episode + 1}.pkl","wb") as fi:
                pkl.dump(Q_table, fi)

    cv2.destroyAllWindows()
    # return Q_table, episode_rewards
    return Q_table


def update_qvalue(Q_table, action_space, obs, a_list, r_list, obs_prime, no_of_agents):
    for i in range(no_of_agents):
        a = a_list[i]
        r = r_list[i]
        if obs[f"A_{i}"][1] == -1:
            s_prime = get_state_representation(obs_prime, i, 'pick')
            s = get_state_representation(obs, i, 'pick')
            if s_prime in Q_table[0].keys():
                q_next = Q_table[0][s_prime]
            else:
                q_next = np.zeros(action_space)
                Q_table[0][s_prime] = q_next
            if s not in Q_table[0].keys():
                Q_table[0][s] = np.zeros(action_space)

            Q_table[0][s][a] = Q_table[0][s][a] + learning_rate * (r + discount_factor * np.max(q_next) - Q_table[0][s][a])
        else:
            s_prime = get_state_representation(obs_prime, i, 'drop')
            s = get_state_representation(obs, i, 'drop')
            if s_prime in Q_table[1].keys():
                q_next = Q_table[1][s_prime]
            else:
                q_next = np.zeros(action_space)
                Q_table[1][s_prime] = q_next
            if s not in Q_table[1].keys():
                Q_table[1][s] = np.zeros(action_space)

            Q_table[1][s][a] = Q_table[1][s][a] + learning_rate * (r + discount_factor * np.max(q_next) - Q_table[1][s][a])


def get_action(Q_table, action_space, ep, obs, no_of_agents):
    # if np.random.randn(0, 1) < ep:
    if 0.5 < ep:
        action_list = [random.randint(0, action_space - 1) for i in range(no_of_agents)]

    else:
        action_list = []
        Q_pick = Q_table[0]
        Q_drop = Q_table[1]
        for i in range(no_of_agents):
            if obs[f"A_{i}"][1] == -1:
                s = get_state_representation(obs, i, 'pick')
                if s in Q_pick:
                    where_max = np.where(Q_pick[s] == np.max(Q_pick[s]))[0]
                    if len(where_max) == 1:
                        a = where_max[0]
                    else:
                        a = np.random.choice(where_max)
                    action_list.append(a)
                else:
                    action_list.append(random.randint(0, action_space - 1))
            else:
                s = get_state_representation(obs, i, 'drop')
                if s in Q_drop:
                    where_max = np.where(Q_drop[s] == np.max(Q_drop[s]))[0]
                    if len(where_max) == 1:
                        a = where_max[0]
                    else:
                        a = np.random.choice(where_max)
                    action_list.append(a)
                else:
                    action_list.append(random.randint(0, action_space - 1))
    return action_list


def get_state_representation(obs, i, task):
    rep = np.array2string(obs[f"A_{i}"][0]) + ";"
    if task == 'drop':
        return rep

    j = obs[f"A_{i}"][2]
    if j == -1:
        return " "
    rep += np.array2string(obs[f"Gem_{j}"][0]) + ";"
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


