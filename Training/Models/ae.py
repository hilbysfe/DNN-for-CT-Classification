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
	
	return conv_out, kernel
	
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
		
	# with tf.variable_scope(name + '/visualization'):
			# # scale weights to [0 1], type is still float
			# kernel_avg = tf.reduce_mean(kernel, axis=2)
			# x_min = tf.reduce_min(kernel_avg)
			# x_max = tf.reduce_max(kernel_avg)
			# kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
			
			# # to tf.image_summary format [batch_size, height, width, channels]
			# kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
			# kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
			# batch = kernel_transposed.get_shape()[0].value
						
			# tf.summary.image('/filters', kernel_transposed, max_outputs=batch)			
			
	return conv_out, kernel
		
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

	
def _deconv_layer(input, kshape, oshape, strides, padding, name):
	with tf.variable_scope(name):
		kernel = tf.get_variable(
			'weights',
			kshape,
			initializer=tf.contrib.layers.xavier_initializer(),
			dtype=tf.float32)
		biases = tf.get_variable(
			'biases',
			[kshape[-2]],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)
				
		deconv = tf.nn.conv2d_transpose(input, kernel, oshape, strides, padding)
		deconv = tf.nn.bias_add(deconv, biases)
		
		deconv = tf.nn.relu(deconv, name='Activation')
		_activation_summary(deconv)
		
	# with tf.variable_scope(name + '/visualization'):
			# # scale weights to [0 1], type is still float
			# kernel_avg = tf.reduce_mean(kernel, axis=2)
			# x_min = tf.reduce_min(kernel_avg)
			# x_max = tf.reduce_max(kernel_avg)
			# kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
			
			# # to tf.image_summary format [batch_size, height, width, channels]
			# kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
			# kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
			# batch = kernel_transposed.get_shape()[0].value
						
			# tf.summary.image('/filters', kernel_transposed, max_outputs=batch)			
			
	return deconv, kernel
	
def _deconv_layer2(input, shape, name, ksize=11, stride=1, padding='SAME'):
	W = encoder[layer_i]
	b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
	output = lrelu(tf.add(
		tf.nn.conv2d_transpose(
			current_input, W,
			tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
			strides=[1, 2, 2, 1], padding='SAME'), b))
	current_input = output

		
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

def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))
	
def keep_top_k(x, k):	
	values, indices = tf.nn.top_k(-x, k)
	
	a = np.zeros((x.get_shape()[0].value, x.get_shape()[1].value, x.get_shape()[2].value, len(x.get_shape().as_list())-1))
	for i in range(x.get_shape()[0].value):
		for j in range(x.get_shape()[1].value):
			for h in range(x.get_shape()[2].value):
				a[i,j,h,:] = [i,j,h]
	a = tf.convert_to_tensor(a, dtype=tf.int32)
	
	full_indices = tf.concat(3, [a, tf.expand_dims(indices[:,:,:,0], 3)])
	for i in range(1, k):
		full_indices = tf.concat(3, [full_indices, 
									 tf.concat(3, [a, tf.expand_dims(indices[:,:,:,i], 3)])]
								)
	full_indices2 = tf.reshape(full_indices, [-1, len(x.get_shape().as_list())])
	
	to_substract = tf.sparse_to_dense(full_indices2, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
	
	return x + to_substract
	
def spatial_sparsity(x):
	return keep_top_k(x, 1)
	
def lifetime_sparsity(x, lifetime):
	return tf.transpose(keep_top_k(tf.transpose(x, [3, 1, 2, 0]), lifetime), [3, 0, 1, 2])	
	
def _WTA_layer(input, lifetime='None'):
	net = spatial_sparsity(input)
	
	if lifetime != 'None':
		net = lifetime_sparsity(net, lifetime)
	
	return net
	
def _structured_sparsity_layer(x):
	# first across featuremap
	examp_sparsity = tf.norm(x, ord=2, axis=3, keep_dims=True)
	net = tf.divide(x, examp_sparsity+1e-9)
	# then per featuremap
	net = tf.reshape(net, [net.get_shape()[1].value, net.get_shape()[2].value, -1])
	feat_sparsity = tf.norm(net, ord=2, axis=2, keep_dims=True)
	net = tf.divide(net, feat_sparsity+1e-9)
	# back to initial shape
	net = tf.reshape(net, [-1, x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value])
	
	return net
	
	
class Autoencoder(object):
	
	def __init__(self, network_architecture, learning_rate=0.001, sp_lambda=1):
		self.network_architecture = network_architecture
		self.learning_rate = learning_rate
		self.initializer = tf.contrib.layers.xavier_initializer()
		self.sp_lambda = sp_lambda
		
		# tf Graph input
		self.x = tf.placeholder(tf.float32, [None,32,32,3])
				
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
		print(self.x.get_shape())
		shapes = []
		encoder = []
		self.weights = []
		with tf.variable_scope('Recognition_network'):
			# ==== Encoder ====
			# net = corrupt(self.x)
			net = self.x
			for i, (kernel, map) in enumerate(zip(self.network_architecture['Conv_kernels'],self.network_architecture['Conv_maps'])):
				encoder.append(net)
				net,k = _conv_layer(
						input=net, 
						shape=[kernel, kernel, net.get_shape()[3], map], 
						strides=[1,2,2,1], 
						padding='SAME', 
						name='ConvLayer_'+str(i))						
				# net,k = _rfnn_conv_layer(
						# input=net,
						# ksize,
						# fsize,
						# nrbasis,
						# sigmas,
						# stride,
						# padding,
						# name='ConvLayer_'+str(i))
				self.weights.append(k)
				print(net.get_shape())
		
		
		with tf.variable_scope('Structured_sparsity'):
			self.z = _structured_sparsity_layer(net)
			net = self.z
			print(net.get_shape())
		
		with tf.variable_scope('Generator_network'):
			for i, kernel in enumerate(reversed(self.network_architecture['Conv_kernels'])):
				batch = tf.shape(self.x)
				oshape = tf.stack([batch[0], batch[1], batch[2], batch[3]])
				with tf.variable_scope('/ConvLayer_'+str(i)):
					net,k = _deconv_layer(
							input=net, 
							kshape=[kernel, kernel, encoder[-(i+1)].get_shape()[3].value, net.get_shape()[3].value], 
							oshape=oshape,
							strides=[1,2,2,1], 
							padding='SAME', 
							name='DeconvLayer_'+str(i))
					
					print(net.get_shape())
		
		# with tf.variable_scope('Generator_network'):
			# # ==== Decoder ====
			# for i, kernel in enumerate(reversed(self.network_architecture['Conv_kernels'])):
				# net = _deconv_layer(
						# input=net, 
						# ksize=kernel,
						# shape=shapes[-(i+2)], 
						# stride=2, 
						# padding='SAME', 
						# name='DeconvLayer_'+str(i))
				# print(net.get_shape())			
			
			# # ==== Reconstruction ====					
		self.x_reconstr = net
		_activation_summary(self.x_reconstr)

	def w_psp(self, a):
		ashape = a.get_shape()
		a_vector = tf.reshape(a, [-1, ashape[1].value*ashape[2].value*ashape[3].value])
		
	
	def w_lsp(self, a):
		ashape = a.get_shape()
		a_vector = tf.reshape(a, [-1, ashape[1].value*ashape[2].value*ashape[3].value])
	
			
	def _create_loss_optimizer(self):
		# --- Reconstruction loss ---
		self.reconstr_loss = tf.reduce_mean(tf.square(self.x_reconstr - self.x))
		
		# --- Sparsity loss ---
		self.sparsity_loss = tf.norm(self.z, ord=1, axis=(2, 3))
		self.sparsity_loss = tf.reduce_mean(self.sparsity_loss, axis=(0, 1))
		
		self.cost = self.reconstr_loss + self.sp_lambda * self.sparsity_loss
		
		self.optimizer = \
			tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
		
	def partial_fit(self, X):
		opt, total = self.sess.run((self.optimizer, self.cost), 
								  feed_dict={self.x: X})		
		
		return total
	
		
	def reconstruct(self, X):
		return self.sess.run(self.x_reconstr, 
							 feed_dict={self.x: X})
							 
	def logs(self, X):
		merged = tf.summary.merge_all()
		
		return self.sess.run(merged, feed_dict={self.x: X})
		
	
	def get_kernels(self):			
		kernel = tf.get_default_graph().get_tensor_by_name("Recognition_network/ConvLayer_0/weights:0")
		
		# kernel_avg = tf.reduce_mean(kernel, axis=2)
		
		# to tf.image_summary format [batch_size, height, width, channels]
		kernel_transposed = tf.transpose(kernel, [3, 0, 1, 2])
			
		return self.sess.run(kernel_transposed)

def _activation_summary(x):
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name, x)
							 