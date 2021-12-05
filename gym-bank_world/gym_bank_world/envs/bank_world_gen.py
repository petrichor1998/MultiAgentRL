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


def world_gen(n=10, goal_length=3, no_of_agents = 2):
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
    # locs =  [np.array([0, n-1]), np.array([n-1, 0]), np.array([3, 2]), np.array([5, 3]), np.array([9, 9])]
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