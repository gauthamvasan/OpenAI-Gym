# I'm using Keras for creating the Neural Nets - Heard it's easier to use compared to Theano/TensorFlow
# This implementation follows the Atari Deep RL work (Deep Q Networks) - Unlike their Convolutional neural net, I use a normal multi-layered, feed forward network

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
replay_capacity = 500
input_shape = 4
batch_size = 100
gamma = 0.99
epsilon_q = 1.0
num_actions = 2
numEpisodes = 500
numRuns = 10

# Define the model
model = Sequential()
model.add(Dense(8, input_dim=4,  activation='relu')) # An "activation" is just a non-linear function applied to the output
# of the layer above. Here, with a "rectified linear unit",
# we clamp all values below 0 to 0.
model.add(Dense(16,  activation='relu'))
#model.add(Dropout(0.2))# Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(num_actions, activation='linear')) # This special "softmax" activation among other things,
# ensures the output is a valid probability distribution, that is
# that its values are all non-negative and sum to 1.


#model.compile(loss='mean_squared_error', optimizer=SGD(lr=math.pow(10,-4), momentum=0.9, nesterov=True))
model.compile(loss='mse', optimizer='adam')

class replay_memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.sample_count = 0
        self.index = 0
        self.full = False
        self.state1_memory = np.zeros((capacity, input_shape))
        self.action_memory = np.zeros(capacity, dtype=np.uint8)
        self.reward_memory = np.zeros(capacity, dtype=np.uint8)
        self.done_memory = np.zeros(capacity, dtype=np.uint8)
        self.greedy_memory = np.zeros(capacity, dtype=np.uint8)
        self.state2_memory = np.zeros((capacity, input_shape))

    def add_entry(self, state1, action, reward, state2, done, greedy):
        self.state1_memory[self.index, :] = state1
        self.state2_memory[self.index, :] = state2
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done
        self.greedy_memory[self.index] = greedy
        self.index += 1
        if(self.index>=self.capacity):
            self.full = True
            self.index = 0
        if not self.full:
            self.size += 1

    def sample_batch_random(self, size):
        batch = np.random.choice(np.arange(0,self.size), size=size)
        states1 = self.state1_memory[batch]
        states2 = self.state2_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        done = self.done_memory[batch]
        greedy = self.greedy_memory[batch]
        return (states1, actions, rewards, states2, done, greedy)

    def sample_batch(self, batch_size):
        start = self.size - batch_size
        end = self.size
        states1 = self.state1_memory[start:end]
        states2 = self.state2_memory[start:end]
        actions = self.action_memory[start:end]
        rewards = self.reward_memory[start:end]
        done = self.done_memory[start:end]
        greedy = self.greedy_memory[start:end]
        return (states1, actions, rewards, states2, done, greedy)


def epsilon_greedy_policy(state):
    rand = random.random()
    scores = model.predict(state.reshape((1,4)))
    greedy = 1
    if (rand < epsilon_q):
        greedy = 0
        index = np.random.randint(num_actions)
        return index, greedy

    else:
        return scores.argmax(), greedy

def Q_update():
    if mem.size > batch_size:
        size = batch_size
    else:
        size = np.random.randint(mem.size) + 1
    states1, actions, rewards, states2, done, greedy = mem.sample_batch_random(size)
    #targets = np.zeros((rewards.size,num_actions))
    old_Q = model.predict(states1)
    new_Q = model.predict(states2)
    targets = np.copy(old_Q)
    for i in range(size):
        if done[i]:
            targets[i,actions[i]] = rewards[i]
        else:
            targets[i,actions[i]] = rewards[i] + gamma * np.max(new_Q[i])
        #print(actions[i], targets[i,actions[i]] - old_Q[i,actions[i]])
    model.train_on_batch(states1,targets)


mem = replay_memory(replay_capacity)

if __name__ == '__main__':
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        t = 0
        returns = 0
        while(1):
            env.render()
            action, greedy = epsilon_greedy_policy(current_state)
            #print(current_state,type(current_state),env.action_space.sample())
            next_state, reward, done, info = env.step(action)
            returns += reward
            mem.add_entry(current_state, action, reward, next_state, done, greedy)
            Q_update()
            current_state = next_state
            t += 1
            mem.sample_count += 1
            if done or returns>200:
                print("Episode " + str(i_episode) + " finished after {} timesteps with {} reward and {} epsilon".format(t+1,returns,epsilon_q))
                break
            if epsilon_q >= 0.1:
                epsilon_q -= 1.0/(numEpisodes)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")