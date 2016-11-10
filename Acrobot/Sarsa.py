import numpy as np

class Sarsa():
    def __init__(self, alpha = 0.1, gamma = 0.99, lmbda = 0.7, n = 2048, num_actions = 3):
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.n = n

        self.w = np.zeros((n,num_actions))
        self.e = np.zeros((n,num_actions))




