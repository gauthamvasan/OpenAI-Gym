import random
import numpy as np
import tensorflow as tf

class actorCritic(object):
	def __init__(self, session, optimizer_critic, optimizer_actor, critic_network, actor_network, gamma_lmbda, state_dim, num_actions, summary_writer=None, summary_every=5):

		self.session = session
		self.summary_writer = summary_writer
		self.optimizer_critic = optimizer_critic
		self.optimizer_actor = optimizer_actor
		
		self.actor_network = actor_network
		self.critic_network = critic_network

		self.state_dim = state_dim
		self.num_actions = num_actions
		self.gamma_lmbda = tf.constant(gamma_lmbda)

		# initialize the graph on tensorflow
		self.create_variables()
		var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
		self.session.run(tf.initialize_variables(var_lists))

		# make sure the variables in graph are initialized
		self.session.run(tf.assert_variables_initialized())

		if self.summary_writer is not None:
			self.summary_writer.add_graph(self.session.graph)
			self.summary_every = summary_every

	def create_variables(self):
		with tf.name_scope("model_inputs"):
			self.state = tf.placeholder(tf.float32, (None, self.state_dim), name="state")

		with tf.name_scope("predict_actions"):
			with tf.variable_scope("actor_network"):
				self.policy_output = self.actor_network(self.state)
			with tf.variable_scope("critic_network"):
				self.value_output = self.critic_network(self.state)

		self.action_scores = tf.identity(self.policy_output, name="action_scores")

		actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
		critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

		with tf.name_scope("eligibility_traces"):
			self.actor_traces = [tf.Variable(tf.zeros(train_var.get_shape()), trainable=False) for train_var in actor_network_variables]
			self.critic_traces = [tf.Variable(tf.zeros(train_var.get_shape()), trainable=False) for train_var in critic_network_variables]

		with tf.name_scope("compute_pg_gradients"):
			self.action_taken = tf.placeholder(tf.int32, (None, ), name="action_taken")
			# self.lambda_return = tf.placeholder(tf.float32, (None, ), name="lambda_return")
			self.update_target = tf.placeholder(tf.float32, (None, ), name="update_target")

			with tf.variable_scope("actor_network", reuse=True):
				self.logprob = self.actor_network(self.state)
			with tf.variable_scope("critic_network", reuse=True):
				self.critic_estimate = self.critic_network(self.state)

			self.critic_prediction = tf.identity(self.critic_estimate, name="critic_prediction")

			self.actor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logprob, self.action_taken))

			self.actor_gradients = self.optimizer_actor.compute_gradients(self.actor_loss, actor_network_variables)
			# self.advantage = tf.reduce_sum(self.lambda_return - self.critic_estimate)
			# TD error
			self.delta = tf.reduce_sum(self.update_target - self.critic_estimate)

			# self.actor_traces = {}
			for i, (grad, var) in enumerate(self.actor_gradients):
				# if grad is not None:
					# with tf.variable_scope("actor_traces"):
						# actor_trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name="actor_e_trace")
						# actor_trace_update = actor_trace.assign((self.gamma_lmbda * actor_trace) + grad, use_locking=True)
					
					if grad is not None:
						# actor_trace_update = self.actor_traces[i].assign(tf.add(((grad) / self.delta) * tf.slice(tf.reshape(self.logprob, [self.num_actions]), self.action_taken, [1]), tf.mul(self.gamma_lmbda, self.actor_traces[i].value())))
						# self.actor_gradients[i] = (actor_trace_update * self.delta, var)
						actor_trace_update = self.actor_traces[i].assign(tf.add(grad, tf.mul(self.gamma_lmbda, self.actor_traces[i].value())))
						self.actor_gradients[i] = (actor_trace_update * self.delta, var)

			# self.critic_loss = tf.reduce_mean(tf.square(self.update_target - self.critic_estimate))
			self.critic_loss = tf.reduce_mean(tf.square(self.update_target - self.critic_estimate))

			self.critic_gradients = self.optimizer_critic.compute_gradients(self.critic_loss, critic_network_variables)

			for i, (grad, var) in enumerate(self.critic_gradients):
				# if grad is not None:
					# with tf.variable_scope("critic_traces"):
						# critic_trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name="critic_e_trace")
						# critic_trace_update = critic_trace.assign((self.gamma_lmbda * critic_trace) + grad, use_locking=True)

					if grad is not None and self.delta != 0.0:
						# critic_trace_update = self.critic_traces[i].assign(tf.add(grad / self.delta, tf.mul(self.gamma_lmbda, self.critic_traces[i].value())))
						# self.critic_gradients[i] = (critic_trace_update * self.delta, var)
						critic_trace_update = self.critic_traces[i].assign(tf.add(grad / self.delta, tf.mul(self.gamma_lmbda, self.critic_traces[i].value())))
						self.critic_gradients[i] = (critic_trace_update * self.delta, var)
					elif grad is not None and self.delta == 0.0:
						critic_trace_update = self.critic_traces[i].assign(tf.mul(self.gamma_lmbda, self.critic_traces[i].value()))
						self.critic_gradients[i] = (critic_trace_update * self.delta, var)

			# self.gradients = self.actor_gradients + self.critic_gradients
			# for grad, var in self.actor_gradients:
			# 	tf.histogram_summary(var.name, var)
			# 	if grad is not None:
			# 		tf.histogram_summary(var.name + "/gradients", grad)

			with tf.name_scope("train_actor_critic"):
				self.train_op_critic = self.optimizer_critic.apply_gradients(self.critic_gradients)
				self.train_op_actor = self.optimizer_actor.apply_gradients(self.actor_gradients)

		# self.summarize = tf.merge_all_summaries()
		# self.no_op = tf.no_op()

	def sampleAction(self, state):
		def softmax(y):
			max_y = np.amax(y)
			e = np.exp(y - max_y)
			return e / np.sum(e)

		action_scores = self.session.run(self.action_scores, feed_dict={self.state: state})[0]
		action_probs = softmax(action_scores) - 1e-5
		action = np.argmax(np.random.multinomial(1, action_probs))
		return action

	def updateModel(self, state, action, update_target, step):
		# print("state: {}, action: {}, lambda_return: {}".format(state, action, lambda_return))
		_, summary_str = self.session.run([self.train_op_critic, self.train_op_actor], feed_dict={self.state: state, 
																						self.action_taken: action,
																						self.update_target: update_target})
		# self.summary_writer.add_summary(summary_str, step)

	def predictValue(self, state):
		prediction_value = self.session.run(self.critic_prediction, feed_dict={self.state: state})[0]
		return prediction_value