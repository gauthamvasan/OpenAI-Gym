from pylab import *
import numpy as np
import random
import math
from tiles import *
from cartpole import numTilings, n

# Compatible Off-Policy Deterministic Actor-Critic with Gradient Q Learning
class COPDAC_GQ():
    def __init__(self,gamma = 0.97, lmbda = 0.3, alphaTheta = 0.1/numTilings, alphaT = 0.1/numTilings,alphaU = 0.1/numTilings, alphaV = 0.1/numTilings, alphaW = 0.1/numTilings ):
        #Initialize parameters
        self.gamma = gamma
        self.alphaT = alphaT
        self.alphaV = alphaV
        self.alphaU = alphaU
        self.lmbda = lmbda

        #Initialize Weight Vectors
        self.t = np.zeros(n)
        self.u = np.zeros(n)
        self.v = np.zeros(n)
        self.w = np.zeros(n)

        self.delta = 0.0
        self.reward = 0.0
        self.Qvalue = 0.0
        self.nextQvalue = 0.0

    def Softmax_Policy(self):
        # Compute softmax policy
        policy = numpy.dot(self.weights.reshape((features.size, self.numActions)).T, features.ravel())
        policy = numpy.exp(numpy.clip(policy / self.epsilon, -500, 500))
        policy /= policy.sum()
        return policy
        pass
