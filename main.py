import gym
import gym_bank_world
from gym_bank_world.Qlearning import train



if __name__ == "__main__":
    env = gym.make('bank_world-v0')

    Q_table, episode_reward_list =  train(env, action_space=5, num_episodes=1000, q_path="D:\RL_project\QTable_30.pkl")
