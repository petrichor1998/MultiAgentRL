import gym
import gym_bank_world
# from gym_bank_world.Qlearning import train, test
from gym_bank_world.Qlearning_without_options import train, test



if __name__ == "__main__":
    env = gym.make('bank_world-v0')

    # train(env, action_space=5, num_episodes=6000, q_path=None , verbose=False)
    for i in range(10):
        test(env, action_space=5, q_path="QTable_without_options.pkl", verbose=True)
