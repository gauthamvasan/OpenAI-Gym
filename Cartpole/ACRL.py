# I'm using Keras for creating the Neural Nets - Heard it's easier to use compared to Theano/TensorFlow
# This implementation follows the Atari Deep RL work (Deep Q Networks) - Unlike their Convolutional neural net, I use a normal multi-layered, feed forward network

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from actor_critic import *
import gym
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import random
import math
import tensorflow as tf

#Initialize Cartpole environment
env = gym.make('CartPole-v0')

# Initialize memory parameters
num_inputs = 4
batch_size = 100
gamma = 0.99
lmbda = 0.3
epsilon_q = 1.0
num_actions = 2
numEpisodes = 500
numRuns = 10
num_channels = 5
batch_mem = []


hidden1_units_actor = 8
hidden2_units_actor = 16
hidden1_units_critic = 8
hidden2_units_critic = 16
learning_rate_actor = 1e-5
learning_rate_critic = 1e-4

def Actor_Network(x):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1_actor'):
    weights = tf.Variable(
        tf.truncated_normal([num_inputs, hidden1_units_actor],
                            stddev=1.0 / math.sqrt(float(num_inputs))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units_actor]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2_actor'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units_actor, hidden2_units_actor],
                            stddev=1.0 / math.sqrt(float(hidden1_units_actor))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units_actor]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear_actor'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units_actor, num_actions],
                            stddev=1.0 / math.sqrt(float(hidden2_units_actor))),
        name='weights')
    biases = tf.Variable(tf.zeros([num_actions]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def Critic_Network(x):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1_critic'):
    weights = tf.Variable(
        tf.truncated_normal([num_inputs, hidden1_units_critic],
                            stddev=1.0 / math.sqrt(float(num_inputs))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units_critic]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2_critic'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units_critic, hidden2_units_critic],
                            stddev=1.0 / math.sqrt(float(hidden1_units_critic))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units_critic]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear_critic'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units_critic, num_actions],
                            stddev=1.0 / math.sqrt(float(hidden2_units_critic))),
        name='weights')
    biases = tf.Variable(tf.zeros([num_actions]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def AC_Learn():
    states1, states2 = [np.zeros((batch_size,num_inputs)) for _ in xrange(2)]
    actions, rewards,  done =  [np.zeros((batch_size)) for _ in xrange(3)]
    #targets = np.zeros((rewards.size,num_actions))
    for i in range(batch_size):
        states1[i] = batch_mem[i][0]
        actions[i] = batch_mem[i][1]
        rewards[i] = batch_mem[i][2]
        states2[i] = batch_mem[i][3]
        done[i] = batch_mem[i][4]

    old_Q = model.predict(states1)
    new_Q = model.predict(states2)
    targets = np.copy(old_Q)
    for i in range(batch_size):
        if done[i]:
            targets[i,actions[i]] = rewards[i]
        else:
            targets[i,actions[i]] = rewards[i] + gamma * np.max(new_Q[i])
        #print(actions[i], targets[i,actions[i]] - old_Q[i,actions[i]])
    loss = model.train_on_batch(states1,targets)
    print (loss)

def buffer_func(current_state, action, reward, next_state, done):
    if len(batch_mem) == batch_size:
        batch_mem.remove(batch_mem[0])
    batch_mem.append([current_state, action, reward, next_state, done])

if __name__ == '__main__':
    sample_count = 0
    x = tf.placeholder(tf.float32, (None, num_inputs), name="state")
    #actor_network = Actor_Network(x)
    #critic_network = Critic_Network(x)
    optimizer_actor = tf.train.GradientDescentOptimizer(learning_rate_actor)
    optimizer_critic = tf.train.GradientDescentOptimizer(learning_rate_critic)
    acrl = actorCritic(tf.Session(),optimizer_critic,optimizer_actor,Critic_Network,Actor_Network,gamma*lmbda,num_inputs,num_actions)
    print ("Initialization is complete ")
    for i_episode in range(numEpisodes):
        current_state = env.reset()
        t = 0
        returns = 0
        while(1):
            env.render()
            action = np.random.choice(1)
            #print(current_state,type(current_state),env.action_space.sample())
            next_state, reward, done, info = env.step(action)
            returns += reward
            buffer_func(current_state, action, reward, next_state, done)
            current_state = next_state
            t += 1
            sample_count += 1
            if done or returns>200:
                print("Episode " + str(i_episode) + " finished after {} timesteps with {} reward and {} epsilon".format(t+1,returns,epsilon_q))
                break
