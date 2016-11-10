from pylab import *
import numpy as np
import random
import math, gym
from tiles import *
import gym
env = gym.make('Acrobot-v1')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print (observation,reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#Initialize Hashing Tile Coder
numTilings = 8
cTableSize = 8192
cTable = CollisionTable(cTableSize, 'safe')
F = np.zeros(numTilings)
n = cTableSize

