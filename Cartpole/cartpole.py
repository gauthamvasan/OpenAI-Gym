from pylab import *
import numpy as np
import random
import math
from tiles import *
from ACRL import *
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#Initialize Hashing Tile Coder
numTilings = 8
cTableSize = 8192
cTable = CollisionTable(cTableSize, 'safe') 
F = np.zeros(numTilings)
n = cTableSize

env = gym.make('CartPole-v0')
