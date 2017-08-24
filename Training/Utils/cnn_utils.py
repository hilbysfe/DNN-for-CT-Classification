import numpy as np
import tensorflow as tf

CONV_INITIALIZER = tf.contrib.layers.xavier_initializer()
BIAS_INITIALIZER = tf.constant_initializer(0.0)


def _conv_layer_2d(input, shape, strides, padding, is_training, bnorm=False):
	kernel = tf.get_variable(
		'weights',
		shape,
		initializer=CONV_INITIALIZER,
		dtype=tf.float32)
	# No bias when BN
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)
	
	conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding, name='Pre-Activation')
	
	if bnorm:
		conv = batch_norm_wrapper(conv, is_training, True)
	else:
		conv = tf.nn.bias_add(conv, biases)
		
	conv_out = tf.nn.relu(conv, name='Activation')
		
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
	
	if bnorm:
		return conv_out, kernel
	else:
		return conv_out, kernel, biases

def _conv_layer_2d_with_kernel(input, kernel, strides, padding, is_training, name, bnorm=False):
	# No bias when BN
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[kernel.get_shape()[-1]],
			initializer=BIAS_INITIALIZER,
			dtype=tf.float32)
	
	conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding, name='Pre-Activation')
	
	if bnorm:
		conv = batch_norm_wrapper(conv, is_training, True)
	else:
		conv = tf.nn.bias_add(conv, biases)
		
	conv_out = tf.nn.relu(conv, name='Activation')
	
	with tf.variable_scope('Visualization'):
		# scale weights to [0 1], type is still float
		kernel_avg = tf.reduce_mean(kernel, axis=2)
		x_min = tf.reduce_min(kernel_avg)
		x_max = tf.reduce_max(kernel_avg)
		kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
					
		# to tf.image_summary format [batch_size, height, width, channels]
		kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
		kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
		batch = kernel_transposed.get_shape()[0].value
					
		tf.summary.image('Filters', kernel_transposed, max_outputs=batch)
			
	return conv_out
	
def _conv_layer_2d_with_kernel_and_bias(input, kernel, biases, strides, padding, is_training, bnorm=False):

	conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding, name='Pre-Activation')

	# No bias when BN
	if bnorm:
		conv = batch_norm_wrapper(conv, is_training, True)
	else:
		conv = tf.nn.bias_add(conv, biases)
		
	conv_out = tf.nn.relu(conv, name='Activation')
	
	with tf.variable_scope('Visualization'):
		# scale weights to [0 1], type is still float
		kernel_avg = tf.reduce_mean(kernel, axis=2)
		x_min = tf.reduce_min(kernel_avg)
		x_max = tf.reduce_max(kernel_avg)
		kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
					
		# to tf.image_summary format [batch_size, height, width, channels]
		kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
		kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
		batch = kernel_transposed.get_shape()[0].value
					
		tf.summary.image('Filters', kernel_transposed, max_outputs=batch)
			
	return conv_out
		
def _conv_layer_3d(input, shape, strides, padding, is_training, bnorm=False):
	kernel = tf.get_variable(
		'weights',
		shape,
		initializer=CONV_INITIALIZER,
		dtype=tf.float32)
	# No bias when BN
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=BIAS_INITIALIZER,
			dtype=tf.float32)
	
	conv = tf.nn.conv3d(input, kernel, strides=strides, padding=padding)
		
	if bnorm:
		conv = batch_norm_wrapper(conv, is_training, True)
	else:
		conv = tf.nn.bias_add(conv, biases)
		
	conv_out = tf.nn.relu(conv, name='Activation')
	
	return conv_out
		
def _conv_layer_3d_with_kernel(input, kernel, shape, strides, padding, is_training, bnorm=False):
	# No bias when BN
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=BIAS_INITIALIZER,
			dtype=tf.float32)
	
	conv = tf.nn.conv3d(input, kernel, strides=strides, padding=padding)
		
	if bnorm:
		conv = batch_norm_wrapper(conv, is_training, True)
	else:
		conv = tf.nn.bias_add(conv, biases)
		
	conv_out = tf.nn.relu(conv, name='Activation')
	
	return conv_out
		
	
def _deconv_layer_2d(input, kshape, oshape, strides, padding):
	kernel = tf.get_variable(
		'weights',
		kshape,
		initializer=CONV_INITIALIZER,
		dtype=tf.float32)
	biases = tf.get_variable(
		'biases',
		[kshape[-2]],
		initializer=BIAS_INITIALIZER,
		dtype=tf.float32)
			
	deconv = tf.nn.conv2d_transpose(input, kernel, oshape, strides, padding)
	deconv = tf.nn.bias_add(deconv, biases)
	
	deconv = tf.nn.relu(deconv, name='Activation')
		
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
			
	return deconv
	
def _deconv_layer_tied(input, kernel, oshape, strides, padding='SAME'):
	biases = tf.get_variable(
		'biases',
		[ kernel.get_shape().as_list()[2] ],
		initializer=BIAS_INITIALIZER,
		dtype=tf.float32)
	deconv = tf.nn.conv2d_transpose(input, kernel, oshape, strides, padding)
	deconv = tf.nn.bias_add(deconv, biases)
	
	deconv = tf.nn.relu(deconv, name='Activation')
	
	return deconv
	
def _full_layer(input, shape, act, is_training, regularizer, bnorm=False):
	weights = tf.get_variable(
		'weights', 
		shape=shape,
		initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2/shape[-1]),dtype=tf.float32),
		regularizer=regularizer)
	# No bias when BN
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=BIAS_INITIALIZER,
			dtype=tf.float32)
	
	wx = tf.matmul(input, weights)
	if bnorm:
		wx = batch_norm_wrapper(wx, is_training, False)
	else:
		wx = tf.nn.bias_add(wx, biases)
		
	local = act(wx, name="Activation")
		
	return local

def _softmax_layer(input, shape, bnorm=False):
	weights = tf.get_variable(
		'weights', 
		shape=shape,
		initializer=tf.truncated_normal_initializer(stddev=0.04,dtype=tf.float32),
		regularizer=None)
	# No bias when BN				
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=BIAS_INITIALIZER,
			dtype=tf.float32)
		
	wx = tf.matmul(input, weights, name="Activation")
	if bnorm:
		wx = batch_norm_wrapper(wx, is_training, False)
	else:
		wx = tf.nn.bias_add(wx, biases)
		
	return wx
	
def batch_norm_wrapper(inputs, is_training, is_conv, decay = 0.999):
	
	epsilon = 1e-3
	
	scale = tf.get_variable(name='scale', initializer=tf.constant_initializer(1.0), shape=[inputs.get_shape()[-1]], dtype=tf.float32)
	beta = tf.get_variable(name='beta', initializer=tf.constant_initializer(0.0), shape=[inputs.get_shape()[-1]], dtype=tf.float32)
	pop_mean = tf.get_variable(name='pop_mean', initializer=tf.constant_initializer(0.0), shape=[inputs.get_shape()[-1]], trainable=False, dtype=tf.float32)
	pop_var = tf.get_variable(name='pop_var', initializer=tf.constant_initializer(1.0), shape=[inputs.get_shape()[-1]], trainable=False, dtype=tf.float32)

	def train_case():
		if is_conv:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2,3])
		else:
			batch_mean, batch_var = tf.nn.moments(inputs,[0])
			
		train_mean = tf.assign(pop_mean,
							   pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
							  pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, epsilon)
	def test_case():
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, epsilon)
			
	return tf.cond(is_training, train_case, test_case, name='Bnorm')
	
