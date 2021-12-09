import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

y = np.load(sys.argv[1])
dir = os.path.split(sys.argv[1])[0]

x = np.arange(y.shape[0])
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.plot(x, y, 'b',alpha=0.3)

z = moving_average(y,100)
z = np.append(z, [z[-1] for i in range(y.shape[0]-z.shape[0])])
plt.plot(x,z,'r')

plt.show()
