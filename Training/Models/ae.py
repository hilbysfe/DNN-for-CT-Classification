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
	
class Autoencoder(object):
	
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
		print(self.x.get_shape())
		shapes = []
		encoder = []
		with tf.variable_scope('Recognition_network'):
			# ==== Encoder ====
			net = corrupt(self.x)
			for i, (kernel, map) in enumerate(zip(self.network_architecture['Conv_kernels'],self.network_architecture['Conv_maps'])):
				shapes.append(net.get_shape())	
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
				encoder.append(k)
				print(net.get_shape())
			
		self.z = net
		
		for layer_i, shape in enumerate(reversed(shapes)):
			with tf.variable_scope('Generator_network/ConvLayer_'+str(layer_i)):
				# W = tf.get_variable(
							# 'weights',
							# shape=encoder[-(layer_i+1)].get_shape(),
							# initializer=tf.contrib.layers.xavier_initializer(),
							# regularizer=None)
				W = encoder[-(layer_i+1)]
				b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
				output = tf.nn.relu(tf.add(
					tf.nn.conv2d_transpose(
						net, W, shape,
						strides=[1, 2, 2, 1], padding='SAME'), b))
				net = output
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
		self.reconstr_loss = tf.reduce_mean(tf.square(self.x_reconstr - self.x))
		
		# self.sparsity = w_psp(self.z) + w_lsp(self.z)
		
		self.cost = self.reconstr_loss
		
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
							 