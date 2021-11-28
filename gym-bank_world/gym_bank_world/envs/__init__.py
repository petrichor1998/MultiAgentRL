from gym_bank_world.envs.bank_world_env import BankWorldEnv

import gym
from gym.spaces.discrete import Discrete
from gym.spaces import Box, Dict
from gym import error, utils
from gym.utils import seeding
from collections import deque
import numpy as np
from .bank_world_gen import *
from matplotlib import pyplot as plt
import random
import cv2

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
    4: 'stay'
}
class BankWorldEnv(gym.Env):
    """Bankworld representation
    Args:
      n: specify the size of the field (n x n)
      no_of_gems
      num_distractor
      distractor_length
      world: an existing world data. If this is given, use this data.
             If None, generate a new data by calling world_gen() function
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, n=10, no_of_agents = 2, no_of_gems = 3, max_steps=1000, reward_gem=50, step_cost=-1, reward_bank=500, wall_cost = -5):
        self.n = n
        self.no_of_agents = no_of_agents
        self.no_of_gems = no_of_gems
        self.bank_loc = np.array([int((n-1)/2), int((n-1)/2)])
        #Penalties and Rewards
        self.step_cost = step_cost
        self.reward_gem = reward_gem
        self.reward_bank = reward_bank
        self.wall_cost = wall_cost
        self.total_reward = 0
        # World
        self.world = None
        self.obs_dict = None

        #other settings
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))

        # Game initialization
        self.done = False
        self.np_random_seed = None
        self.achieved_goal = None
        self.reset()

        self.num_env_steps = 0
        self.last_frames = deque(maxlen=3)

    def action_map(self, action):
        if action == 0:
            return np.array([-1, 0])
        if action == 1:
            return np.array([1, 0])
        if action == 2:
            return np.array([0, -1])
        if action == 3:
            return np.array([0, 1])
        if action == 4:
            return np.array([0, 0])

    def step(self, action_list):
        self.num_env_steps += 1
        if self.num_env_steps == self.max_steps:
            self.done = True

        #check and change the location of agents

        for i in range(self.no_of_agents):
            #j_gem is the gem that is acquired
            j_gem = self.obs_dict[f"A_{i}"][1]
            gem_same_flag = False
            for j in range(self.no_of_gems):
                if self.obs_dict[f"A_{i}"][0][0] == self.obs_dict[f"Gem_{j}"][0][0] and self.obs_dict[f"A_{i}"][0][1] == self.obs_dict[f"Gem_{j}"][0][1] and j_gem!=j:
                    gem_same_flag = True
            #if agent location is not equal to the gem location then change the color to grid color
            if not gem_same_flag:
                self.world[self.obs_dict[f"A_{i}"][0][0], self.obs_dict[f"A_{i}"][0][1]] = np.array(grid_color)
            # elif j_gem == j:
            #     self.world[self.obs_dict[f"A_{i}"][0][0], self.obs_dict[f"A_{i}"][0][1]] = np.array(grid_color)

            if self.obs_dict[f"A_{i}"][0][0] == self.bank_loc[0] and self.obs_dict[f"A_{i}"][0][1] == self.bank_loc[1]:
                self.world[self.obs_dict[f"A_{i}"][0][0], self.obs_dict[f"A_{i}"][0][1]] = np.array(bank_color)

            temp = self.obs_dict[f"A_{i}"][0] + self.action_map(action_list[i])

            # print(temp)

            wall_flag = True
            if temp[0] >= 0 and temp[0] < self.n and temp[1] >= 0 and temp[1] < self.n:
                self.obs_dict[f"A_{i}"][0] = temp
                wall_flag = False

            if j_gem!=-1:
                self.obs_dict[f"Gem_{j_gem}"][0] = self.obs_dict[f"A_{i}"][0]

            same_flag = False
            for j in range(self.no_of_gems):

                if j_gem!=j and (self.obs_dict[f"A_{i}"][0][0]==self.obs_dict[f"Gem_{j}"][0][0] and self.obs_dict[f"A_{i}"][0][1]==self.obs_dict[f"Gem_{j}"][0][1]):
                    same_flag = True
            # Adding color to the new position of agent. either agent_color or acq_agent_color
            if not same_flag:
                if j_gem==-1:
                    self.world[self.obs_dict[f"A_{i}"][0][0], self.obs_dict[f"A_{i}"][0][1]] = np.array(agent_color)
                else:
                    self.world[self.obs_dict[f"A_{i}"][0][0], self.obs_dict[f"A_{i}"][0][1]] = np.array(acq_agent_color)


        # print(" ")
        # return self.world, self.obs_dict

        #Setting rewards
        for i in range(self.no_of_agents):
            pos_reward = False
            for j in range(self.no_of_gems):
                if self.obs_dict[f"A_{i}"][0][0] == self.obs_dict[f"Gem_{j}"][0][0] and self.obs_dict[f"A_{i}"][0][1] == self.obs_dict[f"Gem_{j}"][0][1]:
                    if self.obs_dict[f"Gem_{j}"][1] == -1 and not wall_flag and self.obs_dict[f"A_{i}"][2] == j:
                        self.obs_dict[f"A_{i}"][1] = j
                        self.obs_dict[f"Gem_{j}"][1] = i
                        self.total_reward += self.reward_gem
                        pos_reward = True

            #Bank Reward!
            if self.obs_dict[f"A_{i}"][1] != -1 and not wall_flag:
                j = self.obs_dict[f"A_{i}"][1]
                if self.obs_dict[f"A_{i}"][0][0] == self.bank_loc[0] and self.obs_dict[f"A_{i}"][0][1] == self.bank_loc[1]:
                    self.obs_dict[f"A_{i}"][1] = -1
                    self.obs_dict[f"A_{i}"][2] = -1
                    self.obs_dict[f"Gem_{j}"][0] = self.bank_loc
                    self.total_reward += self.reward_bank
                    self.planner(i)
                    pos_reward = True
            if not pos_reward:
                if wall_flag:
                    self.total_reward += self.step_cost + self.wall_cost
                else:
                    self.total_reward += self.step_cost

        num_acquired = 0
        for j in range(self.no_of_gems):
            if self.obs_dict[f"Gem_{j}"][0][0] == self.bank_loc[0] and self.obs_dict[f"Gem_{j}"][0][1] == self.bank_loc[1]:
                num_acquired+=1
        if num_acquired==self.no_of_gems:
            self.done = True

        return self.obs_dict, self.total_reward, self.done, {}



    def reset(self):
        self.num_env_steps = 0
        self.total_reward = 0
        self.done = False
        self.world, self.obs_dict = world_gen(self.n, self.no_of_gems, self.no_of_agents)
        for i in range(self.no_of_agents):
            self.planner(i)

    def render(self, mode="human"):
        #
        # plt.clf()
        # plt.imshow(self.world)
        # plt.show()
        img = cv2.resize(self.world, (512, 512))
        cv2.putText(img, str(self.num_env_steps), (400,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("trajectories", img)
        cv2.waitKey(100)

    def dist(self, i, j):
        agent_loc = self.obs_dict[f"A_{i}"][0]
        gem_loc = self.obs_dict[f"Gem_{j}"][0]

        distance = np.sum(np.abs(agent_loc - gem_loc))

        return distance

    def planner(self, agent_id):
        i = agent_id
        dist_list = []

        for j in range(self.no_of_gems):
            if self.obs_dict[f"Gem_{j}"][2] == -1:
                dist_list.append(self.dist(i, j))
            else:
                dist_list.append(np.inf)
        j = np.argmin(dist_list)
        if dist_list[j] == np.inf:
            return False
        else:
            self.obs_dict[f"A_{i}"][2] = j
            self.obs_dict[f"Gem_{j}"][2] = i

        return True

if __name__ == "__main__":
    # import gym

    # execute only if run as a script
    # env = gym.make('bank_world-v0')
    num_agents = 2
    env = BankWorldEnv(20, no_of_agents= num_agents, no_of_gems= 60, max_steps=50, reward_bank=1000)

    # env.render()
    for episode in range(3):
        print ('# Episode:', episode+1)
        env.reset()
        env.render()
        while not env.done:
            action_list = [random.randint(0, 4) for i in range(num_agents)]
            env.step(action_list)
            env.render()
        print (env.total_reward)

    cv2.destroyAllWindows()


import numpy as np
import random
from matplotlib import pyplot as plt

# select the colors for the entire environment
colors = {1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
                       4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
                       9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
                       14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
                       0: [0, 0, 117]}

COLOR_ID = dict([(tuple(v), k) for k, v in colors.items()])  # , "wall"])])

num_colors = len(colors)
agent_color = [0, 255, 0]
acq_agent_color = [0, 0, 255]

#bank will be at the center of the grid
# bank_color = [255, 255, 255]
bank_color = [0, 0, 0]
grid_color = [220, 220, 220]
gem_color = [255, 0, 0]
# Wall will be on the boundary of the grid
wall_color = [0, 0, 0]


def world_gen(n=12, goal_length=3, no_of_agents = 2):
    """generate BankWorld
    """

    # For bank_world keys are gem positions and values are -1 if no agent and A_i if agent i is assigned
    #
    obs_dict = {}
    # Create the grid with gray color
    world = np.ones((n, n, 3), dtype= np.uint8) * 220
    """
        Return the position of Gems  and agents on the grid
    """
    coord_list = []
    bank_loc = np.zeros(2, dtype= np.int)
    for i in range(n):
        for j in range(n):
            if i == int((n - 1)/2) and j == int((n - 1)/2):
                bank_loc[0] = i
                bank_loc[1] = j
            else:
                coord_list.append(np.array([i, j], dtype=np.int))

    #Sampling locs for agents and gems
    locs = random.sample(coord_list, no_of_agents + goal_length)

    # getting agent locs and assignment
    # obs_dict[f"A_{i}"] = [loc, acquired_gem_id/-1, to_acquire_gem_id/-1]
    for i in range(no_of_agents):
        obs_dict[f"A_{i}"] = [locs[i], -1, -1]

    #getting gem locs and assignment
    #obs_dict[f"Gem_{j}"] = [locs[i], acquired_agent_id, to_acquire_agent_id]
    for i in range(no_of_agents, no_of_agents + goal_length):
        obs_dict[f"Gem_{i - no_of_agents}"] = [locs[i], -1, -1]


    """
        Here we place the gems and bank on the grid.       
    """
    #adding bank to the world
    world[bank_loc[0], bank_loc[1]] = np.array(bank_color)

    #adding all locs to world
    for k, v in obs_dict.items():
        world[v[0][0], v[0][1]] = agent_color if "A" in k else gem_color

    return world, obs_dict

if __name__ == "__main__":
    world, obs_dict = world_gen(30)
    plt.imshow(world)
    plt.show()
