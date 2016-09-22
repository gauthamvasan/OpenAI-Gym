import logging
#logging.basicConfig(filename="logs_cartpole/expt7_cartpole.log", level=logging.INFO)

from actor_critic import actorCritic
import tensorflow as tf
import numpy as np
import Queue
import time
from cartpole import CartPole
from joblib import Parallel, delayed

# state and no. of actions in mountaincar
state_dim = 4
num_actions = 2

# pos and vel ranges for mountaincar domain
# x_range = [-0.5, 1.2]
# v_range = [-0.07, 0.07]

# no. of hidden units for neural networks (actor & critic)
num_hid = 50

# alpha_v_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
alpha_v_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
alpha_pi_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# alpha_pi_list = [0.00001, 0.00005]

def rescale_states(x1, x2, x3, x4):
	new_x1 = x1 / 2.4
	new_x2 = x2 / 2.4
	new_x3 = x3 / 0.2094384
	new_x4 = x4 / 0.2094384
	return (new_x1, new_x2, new_x3, new_x4)

def compute_K(gamma, lmbda):
	if(lmbda * gamma) <= 0.01:
		return 1
	elif(lmbda * gamma) >= 1:
		return 1e+7
	else:
		return np.ceil(np.log(0.01) / np.log(gamma * lmbda))

def actor_network(states):
  # define policy neural network
	W1 = tf.get_variable("W1", [state_dim, num_hid],
						initializer=tf.random_normal_initializer())
	b1 = tf.get_variable("b1", [num_hid],
						initializer=tf.constant_initializer(0))
	h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
	W2 = tf.get_variable("W2", [num_hid, num_actions],
						initializer=tf.random_normal_initializer(stddev=0.1))
	b2 = tf.get_variable("b2", [num_actions],
						initializer=tf.constant_initializer(0))
	p = tf.matmul(h1, W2) + b2
	return p

def critic_network(states):
	# define critic neural network
	W1 = tf.get_variable("W1", [state_dim, num_hid],
						initializer=tf.random_normal_initializer())
	b1 = tf.get_variable("b1", [num_hid],
						initializer=tf.constant_initializer(0))
	h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
	W2 = tf.get_variable("W2", [num_hid, 1],
						initializer=tf.random_normal_initializer())
	b2 = tf.get_variable("b2", [1],
						initializer=tf.constant_initializer(0))
	v = tf.matmul(h1, W2) + b2
	return v

def run_expt(alpha_v, alpha_pi, gamma, lmbda):
	# no. of episodes and runs
	num_episodes = 1000
	max_steps = 1000
	
	steps_per_episode = np.zeros((num_episodes, ))
	avg_steps = 0.0

	sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

	optimizer_critic = tf.train.GradientDescentOptimizer(learning_rate=alpha_v)
	optimizer_actor = tf.train.GradientDescentOptimizer(learning_rate=alpha_pi)

	ac_agent = actorCritic(sess, optimizer_critic, optimizer_actor, critic_network, actor_network, gamma * lmbda, state_dim, num_actions)
	cartpole_domain = CartPole()
	for current_episode in range(num_episodes):
		current_state = cartpole_domain.reset()
		rescaled_current_state = rescale_states(current_state[0], current_state[1], current_state[2], current_state[3])

		update_target = np.zeros(1)
		G = 0.0
		step = 0

		while current_state is not None and step < max_steps:
			a_t = ac_agent.sampleAction(np.array(rescaled_current_state).reshape(1, state_dim))
			r_t, next_state = cartpole_domain.move(a_t)

			G += (gamma * r_t)
			step += 1

			# v_current = ac_agent.predictValue(np.array(rescaled_current_state).reshape(1, state_dim))
			v_next = np.zeros(1)
			rescaled_next_state = None
			if next_state is not None:
				rescaled_next_state = rescale_states(next_state[0], next_state[1], next_state[2], next_state[3])
				v_next = ac_agent.predictValue(np.array(rescaled_next_state).reshape(1, state_dim))
				# print("v_next: {}".format(v_next))
			update_target = r_t + (gamma * v_next)
			# print("update_target: {}".format(update_target))
			# print ("update_target: {}".format(update_target))
			# delta = r_t + (gamma * v_next) - v_current
			# print ("delta_prime: {}".format(delta_prime))
			ac_agent.updateModel(np.array(rescaled_current_state).reshape(1, state_dim), np.array([a_t]), np.array(update_target))
			rescaled_current_state = np.copy(rescaled_next_state)
			current_state = next_state

		steps_per_episode[current_episode] = step
		avg_steps = avg_steps + step

	avg_steps = avg_steps * 1.0 / num_episodes
	
	sess.close()
	tf.reset_default_graph()
	
	return (avg_steps, steps_per_episode)

def main():
	num_runs = 50
	num_episodes = 1000

	lmbda = 0.95
	gamma = 1.0

	with Parallel(n_jobs=2) as parallel:
		for alpha_v in alpha_v_list:
			for alpha_pi in alpha_pi_list:
				avg_steps_overall = 0.0
				avg_steps_per_run = np.zeros((num_runs, ))
				avg_steps_per_episode = np.zeros((num_episodes, ))

				start_time = time.clock()

				ret = parallel(delayed(run_expt)(alpha_v, alpha_pi, gamma, lmbda) for current_run in range(num_runs))
				avg_steps_runs, steps_per_episode_runs = zip(*ret)

				# avg_steps_runs = np.zeros((num_runs, ))
				# steps_per_episode_runs = np.zeros((num_runs, ))
				# for current_run in range(num_runs):
				# 	avg_steps_runs[current_run], steps_per_episode_runs[current_run] = run_expt(alpha_v, alpha_pi, gamma, lmbda, K)			

				avg_steps_runs = np.array(avg_steps_runs)  # no. of runs
				steps_per_episode_runs = np.array(steps_per_episode_runs)  # no. of runs x no. of episodes
				avg_steps_overall = np.sum(np.array(avg_steps_runs))
				avg_steps_per_run = np.array(avg_steps_runs)

				end_time = time.clock()

				elapsed_time = (end_time - start_time) / 60.0
				logging.info(' Elapsed time: {}'.format(elapsed_time))

				for run_i in range(num_runs):
					avg_factor = 1.0 / (run_i + 1)
					for episode_i in range(num_episodes):
						avg_steps_per_episode[episode_i] *= (1 - avg_factor)
						avg_steps_per_episode[episode_i] += (avg_factor * steps_per_episode_runs[run_i, episode_i])

				file_name = "plot_cartpole/learning_curve_AC_lambda_" + str(lmbda) + "_a_v_" + str(alpha_v) + "_a_pi_" + str(alpha_pi) + ".npy"
				np.save(file_name, avg_steps_per_episode)

				avg_steps_overall = avg_steps_overall * 1.0 / num_runs
				std_error = 0.0
				for run_i in range(num_runs):
					avg_factor_run = 1.0 / (run_i + 1)
					std_error = ((1 - avg_factor_run) * std_error) + (avg_factor_run * (avg_steps_per_run[run_i] - avg_steps_overall) * (avg_steps_per_run[run_i] - avg_steps_overall))
				std_error = np.sqrt(std_error * 1.0 / num_runs)

				total_steps = avg_steps_overall * num_episodes * num_runs
				logging.info(' alpha_v: {}, alpha_pi: {}'.format(alpha_v, alpha_pi))
				logging.info(' Time per step: {}'.format(elapsed_time * 1.0 / total_steps))
				logging.info(' average reward: {}, std.error: {}'.format(avg_steps_overall, std_error))
				logging.info(' lambda: {}, gamma: {}'.format(lmbda, gamma))
				logging.info(' (nonlinear) Actor-Critic')
				logging.info(' GradientDescentOptimizer-Cartpole domain')

if __name__ == '__main__':
	np.random.seed(1234)
	tf.set_random_seed(1234)
	main()
