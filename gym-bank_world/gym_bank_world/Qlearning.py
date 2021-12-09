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
import copy
random.seed(21)
np.random.seed(21)
# Set learning parameters
learning_rate = .6
discount_factor = 1.0
# num_test_episodes = 100
# log_step = 10000
# total_steps_per_task = 1500000  # 2e6
# test_epsilon = 0.01


def train(train_env, action_space=5, num_episodes=1000, q_path=None, verbose=False):
    # create lists to contain total rewards and steps per episode
    episode_rewards = []
    random_ep = np.arange(0.01, 0.9, 0.99 / (num_episodes), dtype=float)[::-1]
    # random_ep = np.arange(0.01, 0.9, 0.01 , dtype=float)[::-1]
    i = 0

    QPickup = {}
    QDrop = {}
    Q_table = [QPickup, QDrop]
    start_episode = 0
    if q_path is not None:
        start_episode = int(q_path.split('_')[-1].split('.')[0])
        Q_table = pkl.load(open(q_path, "rb"))
        # random_ep = np.arange(0.01, 0.9, 0.99 / (num_episodes - start_episode), dtype=float)[::-1]
    for episode in range(start_episode, num_episodes):
        print (f'# Episode {episode+1}:')

        # Reset environment
        train_env.reset()
        if verbose:
            train_env.render()
        ep = random_ep[min(episode, random_ep.shape[0] - 1)]
        # ep = 1
        i += 1

        # # Get state representation
        # s = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)
        s = copy.deepcopy(train_env.obs_dict)
        num_step = 0
        while not train_env.done:
            # Get action
            a_list = get_action(Q_table, action_space, ep, s, train_env.no_of_agents)

            # take step in env and get reward
            train_env.step(a_list)
            if verbose:
                train_env.render()

            # # Get new state
            # s_prime = get_state_representation(train_env.obs_dict, train_env.no_of_agents, train_env.no_of_gems)
            s_prime = copy.deepcopy(train_env.obs_dict)

            # update Q_table
            update_qvalue(Q_table, action_space, s, a_list, train_env.reward_list, s_prime, train_env.no_of_agents)

            s = copy.deepcopy(s_prime)
            # with open(f"QTable_{num_step + 1}.pkl","wb") as fi:
            #     pkl.dump([Q_table, a_list,train_env.reward_list], fi)
            num_step += 1

        episode_rewards.append(train_env.total_reward)
        print(f"Episode {episode + 1} reward: ", train_env.total_reward)
        plt.clf()
        plt.plot(episode_rewards)
        plt.savefig("plot.png")

    with open(f"QTable_options.pkl","wb") as fi:
        pkl.dump(Q_table, fi)

    np.save("episode_rewards_options.npy", np.array(episode_rewards))
    cv2.destroyAllWindows()
    # return Q_table, episode_rewards
    return Q_table, episode_rewards


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
    if random.random() < ep:
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

def test(test_env, action_space=4, q_path=None, verbose=False):
    # create lists to contain total rewards and steps per episode
    QPickup = {}
    QDrop = {}
    Q_table = [QPickup, QDrop]
    episode_rewards = []
    # start_episode = 0
    if q_path is not None:

        Q_table = pkl.load(open(q_path, "rb"))

    # Reset environment
    test_env.reset()
    if verbose:
        test_env.render()
    s = copy.deepcopy(test_env.obs_dict)

    while not test_env.done:
        # Get action
        a_list = get_action(Q_table, action_space, 0.01, s, test_env.no_of_agents)

        # take step in env and get reward
        test_env.step(a_list)
        if verbose:
            test_env.render()
        s_prime = copy.deepcopy(test_env.obs_dict)
        s = copy.deepcopy(s_prime)
        episode_rewards.append(test_env.total_reward)
    print(test_env.total_reward)
    cv2.destroyAllWindows()

    return Q_table, episode_rewards


