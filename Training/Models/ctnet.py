from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def batch_norm_wrapper(inputs, is_training, is_conv, decay = 0.999):
	
	epsilon = 1e-3
	
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

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
			
	return tf.cond(is_training, train_case, test_case)
		


class CTNET(object):

	def __init__(self, n_classes, kernels, maps, maxpool_kernels, l2=.0, dropout_rate_conv=.0, dropout_rate_hidden=.0, is_training = True, conv3d=False):
	
		self.n_classes				= n_classes
		self.is_training			= is_training
		
		self.dropout_rate_conv		= dropout_rate_conv
		self.dropout_rate_hidden	= dropout_rate_hidden
		
		self.act					= tf.nn.relu
		self.kernels				= kernels
		self.maps					= maps
		self.mp_kernels				= maxpool_kernels
		
		self.initializer			= tf.truncated_normal_initializer(stddev=1e-4,dtype=tf.float32)
		if l2 > 0.0:
			self.regularizer 		= tf.contrib.layers.l2_regularizer(l2)
		else:
			self.regularizer		= None
		
		self.inference				= self.inference_3d if conv3d else self.inference_2d
		
		
	def _conv_layer_2d(self, input, shape, stride, padding, name, bnorm=True):
		with tf.variable_scope(name) as scope:
			kernel = tf.get_variable(
				'weights',
				shape,
				initializer=self.initializer,
				dtype=tf.float32)
			# No bias when BN
			if not bnorm:
				biases = tf.get_variable(
					'biases',
					[shape[-1]],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)
			
			conv = tf.nn.conv2d(input, kernel, strides=stride, padding=padding, name='Pre-Activation')
			
			if bnorm:
				conv = batch_norm_wrapper(conv, self.is_training, True)
			else:
				conv = tf.nn.bias_add(conv, biases)
				
			conv_out = self.act(conv, name='Activation')
			self._activation_summary(conv_out)
			
		with tf.variable_scope(name + '/visualization'):
			# scale weights to [0 1], type is still float
			kernel_avg = tf.reduce_mean(kernel, axis=2)
			x_min = tf.reduce_min(kernel_avg)
			x_max = tf.reduce_max(kernel_avg)
			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
			
			# to tf.image_summary format [batch_size, height, width, channels]
			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
			batch = kernel_transposed.get_shape()[0].value
						
			tf.summary.image('/filters', kernel_transposed, max_outputs=batch)			
		return conv_out
		
	def _conv_layer_3d(self, input, shape, stride, padding, name, bnorm=True):
		with tf.variable_scope(name) as scope:						
			kernel = tf.get_variable(
				'weights',
				shape,
				initializer=self.initializer,
				dtype=tf.float32)
			# No bias when BN
			if not bnorm:
				biases = tf.get_variable(
					'biases',
					[shape[-1]],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)
			
			conv = tf.nn.conv3d(input, kernel, strides=stride, padding=padding)
			
			
			if bnorm:
				conv = batch_norm_wrapper(conv, self.is_training, True)
			else:
				conv = tf.nn.bias_add(conv, biases)
				
			conv_out = self.act(conv, name=scope.name)
			self._activation_summary(conv_out)
			
		return conv_out
		
	def _full_layer(self, input, shape, name, bnorm=False):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights', 
				shape=shape,
				initializer= tf.truncated_normal_initializer(stddev=np.sqrt(2/shape[-1]),dtype=tf.float32),
				regularizer=self.regularizer)
			# No bias when BN
			if not bnorm:
				biases = tf.get_variable(
					'biases',
					[shape[-1]],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)
			
			wx = tf.matmul(input, weights)
			if bnorm:
				wx = batch_norm_wrapper(wx, self.is_training, False)
			else:
				wx = tf.nn.bias_add(wx, biases)
				
			local = self.act(wx, name="Activation")
			
			self._activation_summary(local)
			
		return local
	
	def _softmax_layer(self, input, shape, name, bnorm=False):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights', 
				shape=shape,
				initializer = tf.truncated_normal_initializer(stddev=0.04,dtype=tf.float32),
				regularizer=None)
			# No bias when BN				
			if not bnorm:
				biases = tf.get_variable(
					'biases',
					[shape[-1]],
					initializer=tf.constant_initializer(0.0),
					dtype=tf.float32)
				
			wx = tf.matmul(input, weights, name="Activation")
			if bnorm:
				wx = batch_norm_wrapper(wx, self.is_training, False)
			else:
				wx = tf.nn.bias_add(wx, biases)
				
			self._activation_summary(wx)
			
		return wx

	#---------------------------
	# Model Definition
	#---------------------------

	# ---- Use this for 2D models ----
	def inference_2d(self, X):
		print(X.get_shape())
		ch = X.get_shape()[3].value
				
		# ==== Layer 1 ====				
		net = self._conv_layer_2d(
				input=X,
				shape=[self.kernels[0], self.kernels[0], ch, self.maps[0]],
				stride=[1,1,1,1],
				padding='SAME',
				name='ConvLayer1')
		with tf.variable_scope('MaxPool1'):
			net = tf.nn.max_pool(net, ksize=[1,self.mp_kernels[0],self.mp_kernels[0],1], strides=[1,2,2,1], padding="VALID")
			print(net.get_shape())

		if self.dropout_rate_conv > 0.0:
			keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			net = self._conv_layer_2d(
					input=net,
					shape=[self.kernels[1], self.kernels[1], self.maps[0], self.maps[1]],
					stride=[1,1,1,1],
					padding='SAME',
					name='ConvLayer2')
					
			# net = tf.nn.max_pool(net, ksize=[1,self.mp_kernels[1],self.mp_kernels[1],1], strides=[1,2,2,1], padding="VALID")
			# print(net.get_shape())
			
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			net = self._conv_layer_2d(
					input=net,
					shape=[self.kernels[2], self.kernels[2], self.maps[1], self.n_classes],
					stride=[1,1,1,1],
					padding='SAME',
					name='ConvLayer3')
			with tf.variable_scope('MaxPool3'):
				net = tf.nn.max_pool(net, ksize=[1,self.mp_kernels[2],self.mp_kernels[2],1], strides=[1,2,2,1], padding="VALID")
				print(net.get_shape())
					
			if self.dropout_rate_conv > 0.0:
				keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				net = tf.nn.dropout(net, keep_prob)
		
							
		# ==== AVG Pooling ====		
		k = net.get_shape()[1].value
		net = tf.nn.avg_pool(net, ksize=[1,k,k,1], strides=[1,1,1,1], padding="VALID")
		print(net.get_shape())
		
		# ==== Flatten ====		
		with tf.variable_scope('Flatten'):
			fshape = net.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value
			pyx = tf.reshape(net, [-1, dim])
		print(pyx.get_shape())	
		
		# net = self._full_layer(
			# input = net,
			# shape=(dim, 96),
			# name = 'FullLayer1')
			
		# print(net.get_shape())		

		# if self.dropout_rate_hidden > 0.0:
			# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_hidden, 1)
			# net = tf.nn.dropout(net, keep_prob)
						
		# pyx = self._softmax_layer(
					# input = net,
					# shape=(96, self.n_classes),
					# name = 'SoftmaxLayer')
				
		# print(pyx.get_shape())		
		
		return pyx
		
	def inference_3d(self, X):
		print(X.get_shape())
		# ch = X.get_shape()[3].value
				
		# ==== Layer 1 ====				
		net = self._conv_layer_3d(
				input=X,
				shape=[self.kernels[0], self.kernels[0], self.kernels[1], 1, self.maps[0]],
				stride=[1,1,1,1,1],
				padding='SAME',
				name='ConvLayer1')
		with tf.variable_scope('MaxPool1'):
			net = tf.nn.max_pool3d(net, ksize=[1,self.mp_kernels[0],self.mp_kernels[0],self.mp_kernels[0],1], strides=[1,2,2,2,1], padding="VALID")
			print(net.get_shape())

		if self.dropout_rate_conv > 0.0:
			keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			net = self._conv_layer_3d(
					input=net,
					shape=[self.kernels[2], self.kernels[2], self.kernels[3], self.maps[0], self.maps[1]],
					stride=[1,1,1,1,1],
					padding='SAME',
					name='ConvLayer2')

		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			net = self._conv_layer_3d(
					input=net,
					shape=[self.kernels[4], self.kernels[4], self.kernels[5], self.maps[1], self.maps[2]],
					stride=[1,1,1,1,1],
					padding='SAME',
					name='ConvLayer3')
			with tf.variable_scope('MaxPool3'):
				net = tf.nn.max_pool3d(net, ksize=[1,self.mp_kernels[2],self.mp_kernels[2],self.mp_kernels[2],1], strides=[1,2,2,2,1], padding="VALID")
				print(net.get_shape())
					
			if self.dropout_rate_conv > 0.0:
				keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				net = tf.nn.dropout(net, keep_prob)
			
							
		# ==== Layer 4 ====		
		with tf.variable_scope('Flatten'):
			fshape = net.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value*fshape[4].value
			net = tf.reshape(net, [-1, dim])
		print(net.get_shape())		
		
		net = self._full_layer(
			input = net,
			shape=(dim, 96),
			name = 'FullLayer1')
			
		print(net.get_shape())		

		if self.dropout_rate_hidden > 0.0:
			keep_prob = tf.select(self.is_training, 1-self.dropout_rate_hidden, 1)
			net = tf.nn.dropout(net, keep_prob)
						
		pyx = self._softmax_layer(
					input = net,
					shape=(96, self.n_classes),
					name = 'SoftmaxLayer')
				
		print(pyx.get_shape())		
		
		return pyx
		
	
	
	def _activation_summary(self, x):
		tensor_name = x.op.name
		tf.summary.histogram(tensor_name, x)
		
	def _variable_summaries(self, var, name):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + var.op.name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.scalar_summary('stddev/' + var.op.name, stddev)
			tf.scalar_summary('max/' + var.op.name, tf.reduce_max(var))
			tf.scalar_summary('min/' + var.op.name, tf.reduce_min(var))
			tf.histogram_summary(var.op.name, var)
