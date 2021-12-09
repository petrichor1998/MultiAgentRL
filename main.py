import gym
import gym_bank_world
# from gym_bank_world.Qlearning import train, test
from gym_bank_world.Qlearning_without_options import train, test
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('bank_world-v0')

    # train(env, action_space=5, num_episodes=6000, q_path=None , verbose=False)
    rewards = []
    x = np.arange(1000)
    for i in range(10):
        _, rew = test(env, action_space=5, q_path="QTable_without_options\QTable_without_options.pkl", verbose=False)
        rew = np.append(rew, [rew[-1] for j in range(1000-len(rew))])
        rewards.append(rew)
    
    rewards = np.array(rewards)
    
    min_rewards = []
    max_rewards = []
    avg_rewards = []
    for i in range(1000):
        min_rewards.append(np.min(rewards[:,i]))
        max_rewards.append(np.max(rewards[:,i]))
        avg_rewards.append(np.average(rewards[:,i]))
        
    fig, ax = plt.subplots()
    ax.fill_between(x, min_rewards, max_rewards, facecolor="blue", alpha=0.3)
    ax.plot(x, avg_rewards, "r--")
    plt.show()

