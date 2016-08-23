# I'm using Keras for creating the Neural Nets - Heard it's easier to use compared to Theano/TensorFlow

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

import gym
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random
import math

#Initialize Cartpole environment
env = gym.make('CartPole-v0')

# Initialize memory parameters
input_shape = 4
batch_size = 64
gamma = 1
epsilon_q = 0.9
num_actions = 2

# Define the model
model = Sequential()
model.add(Dense(10, input_dim=4, init='uniform', activation='linear')) # An "activation" is just a non-linear function applied to the output
# of the layer above. Here, with a "rectified linear unit",
# we clamp all values below 0 to 0.
model.add(Dense(8, init='uniform', activation='linear'))
model.add(Dropout(0.2))# Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(num_actions, init='uniform', activation='linear')) # This special "softmax" activation among other things,
# ensures the output is a valid probability distribution, that is
# that its values are all non-negative and sum to 1.


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))

def epsilon_greedy_policy(state):
    rand = np.random.random()
    scores = model.predict(state.reshape((1,4)))
    if (rand < epsilon_q):
        return scores.argmax()
    else:
        index = np.random.choice([0,1])
        return index


if __name__ == '__main__':
    numEpisodes = 2000
    numRuns = 10
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        t = 0
        while(1):
            env.render()
            action = epsilon_greedy_policy(current_state)
            #print(current_state,type(current_state),env.action_space.sample())
            next_state, reward, done, info = env.step(action)
            old_Q = model.predict(current_state.reshape((1,input_shape)))
            new_Q = model.predict(next_state.reshape((1,input_shape)))
            target = np.copy(old_Q)
            target[0,action] = reward + gamma*new_Q.max()
            td = target[0,action] -old_Q[0,action]
            cost = model.train_on_batch(current_state.reshape((1,input_shape)),target)
            print(td,cost,action,old_Q)
            current_state = next_state
            t += 1
            if done:
                print("Episode " + str(i_episode) + " finished after {} timesteps".format(t+1))
                break