from pylab import *
import numpy as np
import random
import math
from tiles import *
import gym
from copy import deepcopy, copy


#Initialize Cartpole environemt
env = gym.make('CartPole-v0')
num_inputs = 4
num_actions = 2

#Initialize Hashing Tile Coder
numTilings = 8
cTableSize = 8192
cTable = CollisionTable(cTableSize, 'safe')
F = np.zeros(numTilings)
n = cTableSize + num_actions


class ACRL():
    def __init__(self,gamma = 1.0, alphaV = 0.1, alphaU = 0.01, lmbda = 0.75):
        self.gamma = gamma
        self.alphaV = alphaV
        self.alphaU = alphaU
        self.lmbda = lmbda

        self.ev = np.zeros(n)
        self.ew = np.zeros(n)

        self.w = np.zeros(n)
        self.u = np.zeros(n)

        self.delta = 0.0
        self.R = 0.0
        self.value = 0.0
        self.nextValue = 0.0

        self.compatibleFeatures = np.zeros(n)

    def Value(self,features):
        Val = 0.0
        for index in features:
            Val += self.w[index]
        self.value = Val

    def Next_Value(self,features):
        Val = 0.0
        for index in features:
            Val += self.w[index]
        self.nextValue = Val

    def Delta(self, reward):
        self.delta = reward - self.value

    def Delta_Update(self):
        self.delta += self.gamma*self.nextValue

    def Trace_Update_Critic(self,features):
        self.ev = self.gamma*self.lmbda*self.ev
        for index in features:
            self.ev[index] += 1

    def Trace_Update_Actor(self):
        self.ew = self.gamma * self.lmbda * self.ew + self.compatibleFeatures

    def Weights_Update_Critic(self):
        self.w += self.alphaV * self.delta * self.ev

    def Weights_Update_Actor(self):
        self.u += self.alphaU * self.delta * self.ew

    def Compatible_Features(self, action, action_prob, sample_features):
        self.compatibleFeatures = np.zeros(n)
        sample_features_bits = np.zeros((n,num_actions))
        add = 0
        for i in range(num_actions):
            if i != action:
                for f in sample_features[i]:
                    sample_features_bits[f,i] = action_prob[i]
                self.compatibleFeatures -= sample_features_bits[:,i]
            else:
                for f in sample_features[i]:
                    sample_features_bits[f,i] = 1
                self.compatibleFeatures += sample_features_bits[:,i]

    def getAction(self,action_prob):
        return np.where(action_prob.cumsum() >= np.random.random())[0][0]

    def Erase_Traces(self):
        self.e_mu = np.zeros(n)
        self.ev = np.zeros(n)
        self.e_sigma = np.zeros(n)

#Initialize Actor - Critic parameters
cart = ACRL(1,0.1/numTilings,0.01/numTilings,0.7)

def loadFeatures(stateVars, featureVector):
    stateVars = stateVars.tolist()
    stateVars[0] += 5.0
    stateVars[1] += 0.5
    stateVars[0] *= 10
    stateVars[1] *= 10

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

def gibbs_action_sampler(state):
    sample_states = []
    sample_features = []
    action_prob = np.zeros(num_actions)  # Action probabilities
    gibbs_den = 0     # gibbs policy denominator
    gibbs_num = []
    for i in range(num_actions):
        original_state = copy(state)
        sample_states.append(original_state.step(i)[0])
        features = loadFeatures(sample_states[i],F)
        sample_features.append(F)
        val = 0
        for f in features:
            val += cart.w[f]
        gibbs_num.append(val)
        gibbs_den += math.exp(val)

    for i in range(num_actions):
        prob = math.exp(gibbs_num[i])/gibbs_den
        action_prob[i] = prob

    return action_prob, sample_states, sample_features


if __name__ == '__main__':
    numEpisodes = 500
    numRuns = 10
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        t = 0
        while 1:
            env.render()
            action_prob, sample_states, sample_features = gibbs_action_sampler(env)
            action =  cart.getAction(action_prob)
            current_features = loadFeatures(current_state,F)
            next_state, reward, done, info = env.step(action)
            next_features = loadFeatures(next_state, F)
            cart.Value(current_features)
            cart.Delta(reward)
            cart.Next_Value(next_features)
            cart.Delta_Update()
            cart.Trace_Update_Critic(current_features)
            cart.Weights_Update_Critic()
            cart.Compatible_Features(action,action_prob,sample_features)
            cart.Trace_Update_Actor()
            cart.Weights_Update_Actor()
            t += 1
            if done or t>=200:
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                break
