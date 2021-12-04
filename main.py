import gym
import gym_bank_world
from gym_bank_world.Qlearning import train


if __name__ == "__main__":
    env = gym.make('bank_world-v0')

    train(env, action_space=4, num_episodes=1000, q_path=None, verbose=False)
