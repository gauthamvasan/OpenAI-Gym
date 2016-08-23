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
mem_buffer = []
replay_capacity = 100000
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


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True))

class replay_memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.index = 0
        self.full = False
        self.state1_memory = np.zeros((capacity, input_shape))
        self.action_memory = np.zeros(capacity, dtype=np.uint8)
        self.reward_memory = np.zeros(capacity, dtype=np.uint8)
        self.done_memory = np.zeros(capacity, dtype=np.uint8)
        self.state2_memory = np.zeros((capacity, input_shape))

    def add_entry(self, state1, action, reward, state2, done):
        self.state1_memory[self.index, :] = state1
        self.state2_memory[self.index, :] = state2
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done
        self.index += 1
        if(self.index>=self.capacity):
            self.full = True
            self.index = 0
        if not self.full:
            self.size += 1

    def sample_batch(self, size):
        batch = np.random.choice(np.arange(0,self.size), size=size)
        states1 = self.state1_memory[batch]
        states2 = self.state2_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        done = self.reward_memory[batch]
        return (states1, actions, rewards, states2, done)


def epsilon_greedy_policy(state):
    rand = np.random.random()
    scores = model.predict(state.reshape((1,4)))
    if (rand < epsilon_q):
        return scores.argmax()
    else:
        index = np.random.choice([0,1])
        return index

def Q_update():
    states1, actions, rewards, states2, done = mem.sample_batch(batch_size)
    #targets = np.zeros((rewards.size,num_actions))
    old_Q = model.predict(states1)
    new_Q = model.predict(states2)
    targets = np.copy(old_Q)
    for i in range(batch_size):
        if done[i]:
            targets[i,actions[i]] = -1
        else:
            targets[i,actions[i]] = rewards[i] + gamma * np.max(new_Q)
    #print(states2.shape, targets.shape)
    model.train_on_batch(states1,targets)


mem = replay_memory(replay_capacity)

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
            mem.add_entry(current_state, action, reward, next_state, done)
            old_Q = model.predict(current_state.reshape((1,4)))
            new_Q = model.predict(next_state.reshape((1,4)))
            td = reward + gamma*new_Q.max() -old_Q.max()
            print(old_Q, reward, td)
            if mem.size > batch_size:
                Q_update()
            current_state = next_state
            t += 1
            if done:
                print("Episode " + str(i_episode) + " finished after {} timesteps".format(t+1))
                break