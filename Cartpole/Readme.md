#Cartpole

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. 
The pendulum starts upright, and the goal is to prevent it from falling over. 
A reward of +1 is provided for every timestep that the pole remains upright. 
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

Run the file qlearn_neural_nets.py to train a neural net from scratch for this setting. The network has 4 inputs, two hidden layers
(with 8 and 16 nodes each) and 2 output nodes (since, we need to calculate Q values for 2 actions). The activations are ReLU for
the hidden layers and Linear for the output layer. 
