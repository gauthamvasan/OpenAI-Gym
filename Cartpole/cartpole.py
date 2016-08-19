import gym
from pylab import *
import numpy as np
import random
import math
from tiles import *
from ACRL import *

#Initialize Hashing Tile Coder
numTilings = 8
cTableSize = 8192
cTable = CollisionTable(cTableSize, 'safe') 
F = np.zeros(numTilings)
n = cTableSize

#Initialize Cartpole environemt
env = gym.make('CartPole-v0')

#Initialize Actor - Critic parameters
cart = ACRL(0.97,0,0.1/numTilings,0.01/numTilings,0.7,n)

def loadFeatures(stateVars, featureVector):
    stateVars = stateVars.tolist()
    stateVars[0] += 1.2
    stateVars[1] += 0.07
    stateVars[0] *= 10
    stateVars[1] *= 100
    
    loadtiles(featureVector, 0, numTilings, cTable, stateVars)
    return featureVector
    '''
    As provided in Rich's explanation
           tiles                   ; a provided array for the tile indices to go into
           starting-element        ; first element of "tiles" to be changed (typically 0)
           num-tilings             ; the number of tilings desired
           memory-size             ; the number of possible tile indices
           floats                  ; a list of real values making up the input vector
           ints)                   ; list of optional inputs to get different hashings
    '''

if __name__ == '__main__':
	numEpisodes = 200
	numRuns = 10  
	for i_episode in range(numEpisodes):
		observation = env.reset()
		for t in range(100):
			env.render()        
			action = env.action_space.sample()
			print(observation,type(observation),env.action_space.sample())
			observation, reward, done, info = env.step(action)
			features = loadFeatures(observation, F)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
