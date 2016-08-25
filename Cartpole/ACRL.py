# The implementation below is for discrete action Actor Critic - We use a softmax/Gibbs distribution for the policy
# For the critic we use Sarsa lambda
# On-policy Actor Critic with discrete actions

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
numTilings = 32
cTableSize = 32768
cTable = CollisionTable(cTableSize, 'safe')
F = np.zeros(numTilings)
n = cTableSize


class ACRL():
    def __init__(self,gamma = 0.99, alphaV = 0.1, alphaU = 0.01, lmbda = 0.7):
        self.gamma = gamma
        self.alphaV = alphaV
        self.alphaU = alphaU
        self.lmbda = lmbda
        self.action = 0

        self.eu = np.zeros((n,num_actions))
        self.ew = np.zeros((n,num_actions))

        self.w = np.zeros((n,num_actions))
        self.u = np.zeros((n,num_actions))

        self.delta = 0.0
        self.q_value = 0.0
        self.q_nextValue = 0.0

        self.compatibleFeatures = np.zeros((n,num_actions))

    def Value(self,features):
        Val = 0.0
        for index in features:
            Val += self.w[index,self.action]
        self.q_value = Val

    def Next_Value(self,features,action):
        Val = 0.0
        for index in features:
            Val += self.w[index,action]
        self.q_nextValue = Val

    def Delta(self, reward):
        self.delta = reward - self.q_value

    def Delta_Update(self):
        self.delta += self.gamma*self.q_nextValue

    def Trace_Update_Critic(self,features):
        #self.ew[:,self.action] = self.gamma*self.lmbda*self.ew[:,self.action]
        self.ew = self.gamma*self.lmbda*self.ew
        for index in features:
            self.ew[index,self.action] += 1

    def Trace_Update_Actor(self):
        self.eu = self.gamma * self.lmbda * self.eu
        self.eu += self.compatibleFeatures

    def Weights_Update_Critic(self):
        #self.w[:,self.action]  += self.alphaV * self.delta * self.ew[:,self.action]
        self.w  += self.alphaV * self.delta * self.ew

    def Weights_Update_Actor(self):
        self.u += self.alphaU * self.delta * self.eu

    def Compatible_Features(self, action_prob, features):
        self.compatibleFeatures = np.zeros((n,num_actions))
        for i in range(num_actions):
            sample_features_bits = np.zeros((n, num_actions))
            if i != self.action:
                for f in features:
                    sample_features_bits[f,i] = action_prob[i]
                self.compatibleFeatures -= sample_features_bits
            else:
                for f in features:
                    sample_features_bits[f,i] = 1
                self.compatibleFeatures += sample_features_bits

    def getAction(self,action_prob):
        self.action = np.where(action_prob.cumsum() >= np.random.random())[0][0]
        return self.action

    def Erase_Traces(self):
        self.e_mu = np.zeros((n,num_actions))
        self.ev = np.zeros((n,num_actions))
        self.e_sigma = np.zeros((n,num_actions))


def sample_action(action_prob):
    return np.where(action_prob.cumsum() >= np.random.random())[0][0]

def loadFeatures(stateVars, featureVector):
    stateVars = stateVars.tolist()
    stateVars[0] += 5.0
    stateVars[2] += 0.5
    stateVars[0] *= 10
    stateVars[2] *= 10

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
    action_prob = np.zeros(num_actions)  # Action probabilities
    gibbs_den = 0     # gibbs policy denominator
    gibbs_num = []
    features = loadFeatures(state, F)
    for i in range(num_actions):
        val = 0
        for f in features:
            val += cart.u[f,i]
        gibbs_num.append(val)
        gibbs_den += math.exp(val)

    for i in range(num_actions):
        prob = (math.exp(gibbs_num[i]))/gibbs_den
        action_prob[i] = prob

    return action_prob, features


#Initialize Actor - Critic parameters
cart = ACRL(1,0.1/numTilings,0.1/numTilings,0.3)

if __name__ == '__main__':
    numEpisodes = 500
    numRuns = 10
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        current_features = loadFeatures(current_state,F)
        t = 0
        while 1:
            env.render()
            action_prob, current_features = gibbs_action_sampler(current_state)
            action =  cart.getAction(action_prob)
            next_state, reward, done, info = env.step(action)
            next_action_prob, next_features = gibbs_action_sampler(next_state)

            cart.Value(current_features)
            cart.Delta(reward)
            cart.Next_Value(next_features, sample_action(next_action_prob))
            cart.Delta_Update()
            cart.Trace_Update_Critic(current_features)
            cart.Weights_Update_Critic()
            cart.Compatible_Features(action_prob, current_features)
            cart.Trace_Update_Actor()
            cart.Weights_Update_Actor()
            #print (action_prob)

            current_state = next_state
            t += 1
            if done or t>=200:
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                break
