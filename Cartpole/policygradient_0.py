import numpy as np
import mountaincar
import random
import time
import logging

num_runs = 200
num_episodes = 50
max_steps = 1000
num_actions = 3
# epsilon = 0.0
lmbda = 0.0
gamma = 1.0


num_tilings = 10
num_x_tiles = 10
num_y_tiles = 10
mem_size = num_x_tiles * num_y_tiles * num_tilings

x_range = [-1.2, 0.5]
y_range = [-0.07, 0.07]

tiling_x_offset = []
tiling_y_offset = []

# alpha_v_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# alpha_pi_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
alpha_v_list = [0.5, 1]
alpha_pi_list = [0.01, 0.05, 0.1, 0.5, 1]

def random_float(low, high):
	return random.random() * (high - low) + low


def generateTilingOffsets():
	x_tile_size = (x_range[1] - x_range[0]) * 1.0 / num_x_tiles
	y_tile_size = (y_range[1] - y_range[0]) * 1.0 / num_y_tiles
	for i in range(num_tilings):
		tiling_x_offset.append(np.random.uniform(0, x_tile_size))
		tiling_y_offset.append(np.random.uniform(0, y_tile_size))
	return


def tilecode(x, y):
	x_size = (x_range[1] - x_range[0]) * 1.0 / (num_x_tiles - 1)
	y_size = (y_range[1] - y_range[0]) * 1.0 / (num_y_tiles - 1)
	tiled_features = []
	for i in range(num_tilings):
		x_f = x + tiling_x_offset[i]
		y_f = y + tiling_y_offset[i]
		fx = int(np.floor((x_f - x_range[0]) * 1.0 / x_size))
		fx = np.minimum(fx, num_x_tiles)
		fy = int(np.floor((y_f - y_range[0]) * 1.0 / y_size))
		fy = np.minimum(fy, num_y_tiles)
		ft = fx + (num_x_tiles * fy) + (i * num_x_tiles * num_y_tiles)
		assert ft >= 0
		assert ft < mem_size
		tiled_features.append(ft)
	return tiled_features


def softmax(w, t = 1.0):
	maxQ = np.amax(w)
	e = np.exp(np.array(w - maxQ) / t)
	dist = e / np.sum(e)
	return dist


def sampleAction(theta, phi):
	probability_actions = np.dot(theta.transpose(), phi)
	probability_actions = softmax(probability_actions)
	a_star = np.argmax(probability_actions)
	# if np.random.rand() < epsilon:
	# 	a_star = np.random.randint(0, num_actions)
	phi_action = np.zeros((mem_size, num_actions))
	phi_action[:, a_star] = phi
	p_1 = np.zeros((mem_size, num_actions))
	p_2 = np.zeros((mem_size, num_actions))
	p_3 = np.zeros((mem_size, num_actions))
	p_1[:, 0] = phi
	p_2[:, 1] = phi
	p_3[:, 2] = phi
	gradient_action = phi_action - (probability_actions[0] * p_1) - (probability_actions[1] * p_2) - (probability_actions[2] * p_3)
	return(a_star, gradient_action)


# def sampleActionQ(theta, phi):
# 	probability_actions = np.dot(theta.transpose(), phi)
# 	a_star = np.argmax(probability_actions)
# 	if np.random.rand() < epsilon:
# 		a_star = np.random.randint(0, num_actions)
# 	return (a_star, np.zeros((mem_size, num_actions)))

def plotWeights(theta, w, current_episode):
	fout = open('weights/Trueonline_critic_' + str(current_episode + 1), 'w')
	steps = 50
	for i in range(steps):
		for j in range(steps):
			F = tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps)
			phi_state = np.zeros((mem_size, ))
			phi_state[F] = 1
			height = np.dot(w.transpose(), phi_state)
			fout.write(repr(height) + ' ')
		fout.write('\n')
	fout.close()

	fout = open('weights/Trueonline_actor_' + str(current_episode + 1), 'w')
	steps = 50
	for i in range(steps):
		for j in range(steps):
			F = tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps)
			phi_state = np.zeros((mem_size, ))
			phi_state[F] = 1
			c = np.dot(w.transpose(), phi_state)
			height = np.amax(c)
			fout.write(repr(height) + ' ')
		fout.write('\n')
	fout.close()


def trueOnlinePolicyGradient():
	# logging.basicConfig(filename='example.log',level=logging.DEBUG)
	for alpha_v in alpha_v_list:
		alpha_v = alpha_v * 1.0 / num_tilings
		for alpha_pi in alpha_pi_list:
			alpha_pi = alpha_pi * 1.0 / num_tilings
			print 'alpha_v: ', alpha_v, ' alpha_pi: ', alpha_pi

			avg_steps_overall = 0.0
			avg_steps_per_run = np.zeros((num_runs, ))
			avg_steps_per_episode = np.zeros((num_episodes, ))

			start_time = time.clock()
			for current_run in range(num_runs):
				logging.debug("Run #:" + str(current_run))
				# print 'Run #:', current_run
				theta = 0.00001 * np.random.randn(mem_size, num_actions)
				w = 0.00001 * np.random.randn(mem_size, )
				# w_old = np.zeros((mem_size, ))
				v_old = 0.0

				steps_per_episode = np.zeros((num_episodes, ))
				avg_steps = 0.0

				for current_episode in range(num_episodes):

					# if (current_episode+1) % 10 == 0:
					# 	plotWeights(theta, w, current_episode)

					G = 0.0
					step = 0

					z_theta = np.zeros((mem_size, num_actions))
					z_theta_old = np.zeros((mem_size, num_actions))
					z_w = np.zeros((mem_size, ))

					(pos, vel) = mountaincar.init()
					phi = np.zeros((mem_size, ))
					tiled_indices = tilecode(pos, vel)
					phi[tiled_indices] = 1
					current_state = (pos, vel)
					(a_star, PG_star) = sampleAction(theta, phi)

					a_prime = 0
					PG_prime = np.zeros((mem_size, num_actions))

					while current_state is not None and step < max_steps:
						reward, next_state = mountaincar.sample(current_state, a_star)

						G += (gamma * reward)
						step += 1

						v_current = np.dot(w.transpose(), phi)
						v_next = 0.0
						phi_prime = np.zeros((mem_size, ))
						if next_state is not None:
							tiled_indices = tilecode(next_state[0], next_state[1])
							phi_prime[tiled_indices] = 1
							v_next = np.dot(w.transpose(), phi_prime)
							(a_prime, PG_prime) = sampleAction(theta, phi_prime)
						delta = reward + (gamma * v_next) - v_current

						# z_w = (gamma * lmbda * z_w) + phi - (alpha_v * gamma * lmbda * np.dot(z_w.transpose(), phi) * phi)
						# w += (alpha_v * (delta + v_current - v_old) * z_w - alpha_v * (v_current - v_old) * phi)

						z_w = (gamma * lmbda * z_w) + phi
						w += (alpha_v * delta * z_w)

						# z_theta = (gamma * lmbda * z_theta) + PG_star
						# theta += ((alpha_pi * z_theta * delta) + ((alpha_pi * z_theta_old) * (v_current - v_old)))

						z_theta = (gamma * lmbda * z_theta) + PG_star
						theta += (alpha_pi * delta * z_theta)

						v_old = v_next
						z_theta_old = np.copy(z_theta)
						phi = np.copy(phi_prime)
						a_star = a_prime
						current_state = next_state
						PG_star = np.copy(PG_prime)

					# print '########### Episode: ', current_episode, ' Return: ', G, ' Steps: ', step, " Run: ", current_run
					steps_per_episode[current_episode] = step
					avg_steps += step
				avg_steps = avg_steps * 1.0 / num_episodes
				avg_steps_overall += avg_steps
				avg_steps_per_run[current_run] = avg_steps

				avg_factor = 1.0 / (current_run + 1)
				for episode_i in range(num_episodes):
					avg_steps_per_episode[episode_i] *= (1 - avg_factor)
					avg_steps_per_episode[episode_i] += (avg_factor * steps_per_episode[episode_i])

			end_time = time.clock()
			elapsed_time = (end_time - start_time) / 60.0
			print 'Elapsed time: ', elapsed_time
			# logging.debug('Elapsed time: ' + str(elapsed_time))
			avg_steps_overall = avg_steps_overall * 1.0 / num_runs
			std_error = 0.0
			for run_i in range(num_runs):
				avg_factor_run = 1.0 / (run_i + 1)
				std_error = ((1 - avg_factor_run) * std_error) + (avg_factor_run * (avg_steps_per_run[run_i] - avg_steps_overall) * (avg_steps_per_run[run_i] - avg_steps_overall))
			std_error = np.sqrt(std_error * 1.0 / num_runs)

			total_steps = avg_steps_overall * num_episodes * num_runs
			print 'Time per step: ', (elapsed_time * 1.0 / total_steps)
			print 'alpha_v: ', alpha_v, ' alpha_pi: ', alpha_pi, ' lmbda: ', lmbda
			print  'average reward: ', -1.0 * avg_steps_overall, ' std. error: ', std_error
			print 'Policy gradient'
			# logging.debug('Time per step: ' + str((elapsed_time * 1.0 / total_steps)))
			# logging.debug('alpha_v: ' + str(alpha_v) + ' alpha_pi: ' + str(alpha_pi) + ' lambda: ' + str(lmbda))
			# logging.debug('Average reward: ' + str(-1.0 * avg_steps_overall) + ' std. error: ' + str(std_error))
			# logging.debug('Policy gradient')

			# l_curve = file("learning_curve_policygradient.fp", "wb")
			# np.save(l_curve, avg_steps_per_episode)
			# l_curve.close()

def main():
	random.seed(1234)
	np.random.seed(1234)
	generateTilingOffsets()
	trueOnlinePolicyGradient()

if __name__ == '__main__':
	main()
