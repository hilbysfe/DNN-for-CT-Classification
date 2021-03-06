import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import convolve1d as conv

	
def _rfnn_conv_layer(input, ksize, fsize, nrbasis, sigmas, stride, padding, name):
	with tf.variable_scope(name) as scope:
		basis = init_basis_hermite_2D(ksize[0], sigmas, nrbasis)
		alphas = tf.get_variable(
					name,
					shape=[fsize[1], fsize[0], nrbasis],
					initializer=tf.random_uniform_initializer(-1.0, 1.0))
		# Assuming 1 scale only for now
		kernel = tf.reduce_sum(
				tf.transpose(alphas[:,:,:,None,None]) *
				tf.transpose(basis[None,None,0,:,:,:])
					,axis=2
			)
		biases = tf.get_variable(
			'biases',
			[fsize[-1]],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)
		
		conv = tf.nn.conv2d(input, kernel, strides=stride, padding=padding)
		conv = tf.nn.bias_add(conv, biases)
			
		conv_out = tf.nn.relu(conv, name='Activation')
		_activation_summary(conv_out)
	
	return conv_out
	
def _conv_layer(input, shape, strides, padding, name):
	with tf.variable_scope(name) as scope:
		kernel = tf.get_variable(
			'weights',
			shape,
			initializer=tf.contrib.layers.xavier_initializer(),
			dtype=tf.float32)
		biases = tf.get_variable(
				'biases',
				[shape[-1]],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32)
		
		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
		conv = tf.nn.bias_add(conv, biases)
			
		conv_out = tf.nn.relu(conv, name='Activation')
		_activation_summary(conv_out)
	
	return conv_out
		
def _full_layer(input, shape, name, activation=tf.nn.relu):
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable(
			'weights', 
			shape=shape,
			initializer=tf.contrib.layers.xavier_initializer(),
			regularizer=None)
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)
		
		wx = tf.nn.bias_add(tf.matmul(input, weights), biases)
		local = activation(wx, name="Activation")
		
		_activation_summary(local)
		
	return local
	
def get_deconv_filter(f_shape):
	width = f_shape[0]
	heigh = f_shape[0]
	f = np.ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(heigh):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear

	init = tf.constant_initializer(value=weights, dtype=tf.float32)
			
	return tf.get_variable('deconv_weights', initializer=init, shape=weights.shape)

	
def _deconv_layer(input, shape, name, ksize=11, stride=1, padding='SAME'):
	strides = [1, stride, stride, 1]
	with tf.variable_scope(name):
		in_features = input.get_shape()[3].value

		output_shape = tf.stack(shape)
		f_shape = [ksize, ksize, shape[3], in_features]

		num_input = ksize * ksize * in_features / stride

		weights = get_deconv_filter(f_shape)
		deconv = tf.nn.conv2d_transpose(input, weights, output_shape, strides, padding)

		_activation_summary(deconv)
	
	return deconv

		
def floatX(X):
	return np.asarray(X, dtype=np.float32)

def init_basis_hermite_2D(kernel, sigmas, bases):
	nrBasis = 15
	hermiteBasis = np.empty( (np.int(np.shape(sigmas)[0]), np.int(nrBasis), np.int(kernel), np.int(kernel)) )
	
	for i, sigma in enumerate(sigmas):
		x = np.arange(-np.int((kernel-1)/2), np.int((kernel-1)/2)+1, dtype=np.float)
		impulse = np.zeros( (kernel, kernel) )
		impulse[np.int((kernel-1)/2),np.int((kernel-1)/2)] = 1.0
		
		g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
		g = g/g.sum()
		g1 = sigma * -(x/ np.square(sigma)) * g
		g2 = np.square(sigma) * ( (np.square(x)-np.power(sigma,2)) / np.power(sigma,4)) * g
		g3 = np.power(sigma,3) * -( (np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
		g4 = np.power(sigma,4) * ( ( (np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8) ) ) * g
		gauss0x = conv(impulse, g, axis=1)
		gauss0y = conv(impulse, g, axis=0)
		gauss1x = conv(impulse, g1, axis=1)
		gauss1y = conv(impulse, g1, axis=0)
		gauss2x = conv(impulse, g2, axis=1)
		gauss0 = conv(gauss0x, g, axis=0)
		
		hermiteBasis[i,0,:,:] = gauss0					# g
		hermiteBasis[i,1,:,:] = conv(gauss0y, g1, axis=1) # g_x
		hermiteBasis[i,2,:,:] = conv(gauss0x, g1, axis=0) # g_y
		hermiteBasis[i,3,:,:] = conv(gauss0y, g2, axis=1) # g_xx
		hermiteBasis[i,4,:,:] = conv(gauss0x, g2, axis=0) # g_yy
		hermiteBasis[i,5,:,:] = conv(gauss1x, g1, axis=0) # g_xy
		hermiteBasis[i,6,:,:] = conv(gauss0y, g3, axis=1) # g_xxx
		hermiteBasis[i,7,:,:] = conv(gauss0x, g3, axis=0) # g_yyy
		hermiteBasis[i,8,:,:] = conv(gauss1y, g2, axis=1) # g_xxy
		hermiteBasis[i,9,:,:] = conv(gauss1x, g2, axis=0) # g_yyx
		hermiteBasis[i,10,:,:] = conv(gauss0y, g4, axis=1) # g_xxxx
		hermiteBasis[i,11,:,:] = conv(gauss0x, g4, axis=0) # g_yyyy
		hermiteBasis[i,12,:,:] = conv(gauss1y, g3, axis=1) # g_xxxy
		hermiteBasis[i,13,:,:] = conv(gauss1x, g3, axis=0) # g_yyyx
		hermiteBasis[i,14,:,:] = conv(gauss2x, g2, axis=0) # g_yyxx

	return tf.constant(floatX(hermiteBasis[:,0:bases,:,:]))


class VariationalAutoencoder(object):
	""" Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
	
	This implementation uses probabilistic encoders and decoders using Gaussian 
	distributions and  realized by multi-layer perceptrons. The VAE can be learned
	end-to-end.
	
	See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
	"""
	def __init__(self, network_architecture, learning_rate=0.001, batch_size=100):
		self.network_architecture = network_architecture
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.initializer = tf.contrib.layers.xavier_initializer()
		
		# tf Graph input
		self.x = tf.placeholder(tf.float32, network_architecture['Input'])
		
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
		# Initialize autoencoder network weights and biases
		print(self.x.get_shape())
		shapes = [self.x.get_shape()]
		with tf.variable_scope('Recognition_network'):
			# ==== Recognition ====
			net = self.x
			for i, (kernel, map) in enumerate(zip(self.network_architecture['Conv_kernels'],self.network_architecture['Conv_maps'])):
				net = _conv_layer(
						input=net, 
						shape=[kernel, kernel, net.get_shape()[3], map], 
						strides=[1,2,2,1], 
						padding='SAME', 
						name='ConvLayer_'+str(i))
				shapes.append(net.get_shape())				
				print(net.get_shape())
			
			# ==== Flatten ====
			fshape = net.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value
			net = tf.reshape(net, [-1, dim])
			print(net.get_shape())			

			# ==== Fully Connected 2a ====
			self.z_mean = _full_layer(
							input=net,
							shape=[dim, self.network_architecture['Latent']],
							name='Z_mean')
			_activation_summary(self.z_mean)
			print(self.z_mean.get_shape())
			# ==== Fully Connected 2b ====				
			self.z_log_sigma_sq = _full_layer(
							input=net,
							shape=[dim, self.network_architecture['Latent']],
							name='Z_sigma')
			_activation_summary(self.z_log_sigma_sq)
			print(self.z_log_sigma_sq.get_shape())
		
		with tf.variable_scope('Latent_distribution'):
			# Draw one sample z from Gaussian distribution
			eps = tf.random_normal((self.batch_size, self.network_architecture['Latent']), 0, 1, 
								   dtype=tf.float32)
			# z = mu + sigma*epsilon
			self.z = tf.add(self.z_mean, 
							tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps), name='Z')
			_activation_summary(self.z)
			print(self.z.get_shape())

		with tf.variable_scope('Generator_network'):
			# ==== Fully Connected 1 ====
			net = _full_layer(
						input=self.z,
						shape=[self.network_architecture['Latent'], dim],
						name='ReconLayer1')
			print(net.get_shape())
			
			# ==== Reshape ====
			fshape = shapes[-1]
			net = tf.reshape(net, fshape)
			print(net.get_shape())
			
			for i, kernel in enumerate(reversed(self.network_architecture['Conv_kernels'])):
				net = _deconv_layer(
						input=net, 
						ksize=kernel,
						shape=shapes[-(i+2)], 
						stride=2, 
						padding='SAME', 
						name='DeconvLayer_'+str(i))
				print(net.get_shape())			
			
			# ==== Reconstruction ====					
			self.x_reconstr_mean = tf.nn.sigmoid(tf.reshape(net, [-1, self.network_architecture['Output']]), name='Reconstruction_mean')
			_activation_summary(self.x_reconstr_mean)
			print(self.x_reconstr_mean.get_shape())

			
	def _create_loss_optimizer(self):
		# Adding 1e-10 to avoid evaluation of log(0.0)
		fshape = self.x.get_shape()
		flatten = tf.reshape(self.x, [-1, fshape[1].value*fshape[2].value*fshape[3].value])
		reconstr_loss = \
			-tf.reduce_sum(flatten * tf.log(1e-10 + self.x_reconstr_mean)
						   + (1-flatten) * tf.log(1e-10 + 1 - self.x_reconstr_mean),	1)
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

def _activation_summary(x):
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name, x)
							 