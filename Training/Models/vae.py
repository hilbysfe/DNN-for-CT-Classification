import numpy as np
import tensorflow as tf



class VariationalAutoencoder(object):
	""" Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	
	This implementation uses probabilistic encoders and decoders using Gaussian 
	distributions and  realized by multi-layer perceptrons. The VAE can be learned
	end-to-end.
	
	See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
	"""
	def __init__(self, network_architecture, transfer_fct=tf.nn.elu, 
				 learning_rate=0.001, batch_size=100):
		self.network_architecture = network_architecture
		self.transfer_fct = transfer_fct
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.initializer = tf.contrib.layers.xavier_initializer()
		
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
		
		# Create autoencoder network
		self._create_network()
		# Define loss function based variational upper-bound and corresponding optimizer
		self._create_loss_optimizer()
		
		# Initializing the tensor flow variables
		init = tf.global_variables_initializer()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)
	
	def _create_network(self):
		# Initialize autoencode network weights and biases
		network_weights = self._initialize_weights(**self.network_architecture)

		# Use recognition network to determine mean and 
		# (log) variance of Gaussian distribution in latent
		# space
		with tf.variable_scope('Recognition_network'):
			self.z_mean, self.z_log_sigma_sq = \
				self._recognition_network(network_weights["weights_recog"], 
										  network_weights["biases_recog"])

		with tf.variable_scope('Latent_distribution'):
			# Draw one sample z from Gaussian distribution
			n_z = self.network_architecture["n_z"]
			eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
								   dtype=tf.float32)
			# z = mu + sigma*epsilon
			self.z = tf.add(self.z_mean, 
							tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps), name='Z')
			self._activation_summary(self.z)

		with tf.variable_scope('Generator_network'):
			# Use generator to determine mean of
			# Bernoulli distribution of reconstructed input
			self.x_reconstr_mean = \
				self._generator_network(network_weights["weights_gener"],
										network_weights["biases_gener"])
			
	def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
							n_hidden_gener_1,  n_hidden_gener_2, 
							n_input, n_z):
		all_weights = dict()
		with tf.variable_scope('Encoder'):
			with tf.variable_scope('weights'):
				all_weights['weights_recog'] = {
					'h1': tf.get_variable('h1', shape=(n_input, n_hidden_recog_1), initializer=self.initializer),
					'h2': tf.get_variable('h2', shape=(n_hidden_recog_1, n_hidden_recog_2), initializer=self.initializer),
					'out_mean': tf.get_variable('out_mean', shape=(n_hidden_recog_2, n_z), initializer=self.initializer),
					'out_log_sigma': tf.get_variable('out_log_sigma', shape=(n_hidden_recog_2, n_z), initializer=self.initializer)}
			with tf.variable_scope('biases'):
				all_weights['biases_recog'] = {
					'b1': tf.get_variable('b1', shape=(n_hidden_recog_1), initializer=tf.constant_initializer(0.0)),
					'b2': tf.get_variable('b2', shape=(n_hidden_recog_2), initializer=tf.constant_initializer(0.0)),
					'out_mean': tf.get_variable('out_mean', shape=(n_z), initializer=tf.constant_initializer(0.0)),
					'out_log_sigma': tf.get_variable('out_log_sigma', shape=(n_z), initializer=tf.constant_initializer(0.0))}
		with tf.variable_scope('Decoder'):
			with tf.variable_scope('weights'):
				all_weights['weights_gener'] = {
					'h1': tf.get_variable('h1', shape=(n_z, n_hidden_gener_1), initializer=self.initializer),
					'h2': tf.get_variable('h2', shape=(n_hidden_gener_1, n_hidden_gener_2), initializer=self.initializer),
					'out_mean': tf.get_variable('out_mean', shape=(n_hidden_gener_2, n_input), initializer=self.initializer),
					'out_log_sigma': tf.get_variable('out_log_sigma', shape=(n_hidden_gener_2, n_input), initializer=self.initializer)}
			with tf.variable_scope('biases'):
				all_weights['biases_gener'] = {
					'b1': tf.get_variable('b1', shape=(n_hidden_gener_1), initializer=tf.constant_initializer(0.0)),
					'b2': tf.get_variable('b2', shape=(n_hidden_gener_2), initializer=tf.constant_initializer(0.0)),
					'out_mean': tf.get_variable('out_mean', shape=(n_input), initializer=tf.constant_initializer(0.0)),
					'out_log_sigma': tf.get_variable('out_log_sigma', shape=(n_input), initializer=tf.constant_initializer(0.0))}
		return all_weights
			
	def _recognition_network(self, weights, biases):
		self.layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
										   biases['b1']), name='Layer1') 
		self.layer_2 = self.transfer_fct(tf.add(tf.matmul(self.layer_1, weights['h2']), 
										   biases['b2']), name='Layer2') 
		z_mean = tf.add(tf.matmul(self.layer_2, weights['out_mean']),
						biases['out_mean'], name='Z_mean')
		self._activation_summary(z_mean)
						
		z_log_sigma_sq = \
			tf.add(tf.matmul(self.layer_2, weights['out_log_sigma']), 
				   biases['out_log_sigma'], name='Z_sigma')
		self._activation_summary(z_log_sigma_sq)
		
		return (z_mean, z_log_sigma_sq)

	def _generator_network(self, weights, biases):
		layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
										   biases['b1']), name='Layer1') 
		layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
										   biases['b2']), name='Layer2') 
		x_reconstr_mean = \
			tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
								 biases['out_mean']), name='Reconstruction_mean')
		self._activation_summary(x_reconstr_mean)
								 
		return x_reconstr_mean
			
	def _create_loss_optimizer(self):
		# Adding 1e-10 to avoid evaluation of log(0.0)
		reconstr_loss = \
			-tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
						   + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),	1)
		# tf.summary.scalar('Reconstruction Loss', tf.reduce_mean(reconstr_loss))
		
		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
										   - tf.square(self.z_mean) 
										   - tf.exp(self.z_log_sigma_sq), 1)
		# tf.summary.scalar('KL Divergence', tf.reduce_mean(latent_loss))
		
		self.latent_loss = tf.reduce_mean(latent_loss)
		self.reconstr_loss = tf.reduce_mean(reconstr_loss)
		self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
		
		self.optimizer = \
			tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		
	def partial_fit(self, X):
		"""Train model based on mini-batch of input data.
		
		Return cost of mini-batch.
		"""
		mean, opt, total, latent, recon = self.sess.run((self.x, self.optimizer, self.cost, self.latent_loss, self.reconstr_loss), 
								  feed_dict={self.x: X})		
		# print(np.mean(X))
		
		return total, latent, recon
	
	def transform(self, X):
		"""Transform data by mapping it into the latent space."""
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.sess.run(self.z_mean, feed_dict={self.x: X})
	
	def generate(self, z_mu=None):
		""" Generate data by sampling from latent space.
		
		If z_mu is not None, data for this point in latent space is
		generated. Otherwise, z_mu is drawn from prior in latent 
		space.		
		"""
		if z_mu is None:
			z_mu = np.random.normal(size=self.network_architecture["n_z"])
		# Note: This maps to mean of distribution, we could alternatively
		# sample from Gaussian distribution
		return self.sess.run(self.x_reconstr_mean, 
							 feed_dict={self.z: z_mu})
	
	def reconstruct(self, X):
		""" Use VAE to reconstruct given data. """
		return self.sess.run(self.x_reconstr_mean, 
							 feed_dict={self.x: X})
							 
	def logs(self, X):
		merged = tf.summary.merge_all()
		
		return self.sess.run(merged, feed_dict={self.x: X})

	def _activation_summary(self, x):
		tensor_name = x.op.name
		tf.summary.histogram(tensor_name, x)
							 