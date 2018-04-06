import numpy as np
import tensorflow as tf

from Utils.rfnn_utils import init_basis_hermite_3D
from Utils.rfnn_utils import init_basis_hermite_3D_steerable

# single scale/orientation
from Utils.rfnn_utils import _rfnn_conv_layer_pure_2d


# multi-scale/orientation keeping output
from Utils.rfnn_utils import _rfnn_conv_layer_pure_3d
from Utils.rfnn_utils import _rfnn_conv_layer_pure_3d_SO_learn_sq_bc

from Utils.cnn_utils import _conv_layer_pure_3d

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

class RFNNDenseNet3D(object):
	def __init__(self, growth_rate, depth,
				 total_blocks, keep_prob,
				 model_type,
				 is_training,
				 init_kernel,
				 comp_kernel,
				 init_sigmas,
				 comp_sigmas,
				 init_order,
				 comp_order,
				 thetas,
				 phis,
				 rfnn,
				 bnorm_momentum,
				 renorm=0.7,
				 reduction=1.0,
				 bc_mode=False,
				 beta_wd=0.0,
				 avgpool_kernel_ratio=0.5,
				 avgpool_stride_ratio=0.5,
				 n_classes=10):

		self.kernels = []
		self.alphas = []
		self.conv_act = []
#		self.bc_conv_act = []
#		self.bc_weights = []
		self.weights = []
		self.fl_act = []
		self.bnorm_momentum = bnorm_momentum
		self.renorm = renorm
		self.beta_wd = beta_wd

		self.n_classes = n_classes
		self.depth = depth
		self.growth_rate = growth_rate
		self.bc_mode = bc_mode
		self.avgpool_kernel_ratio = avgpool_kernel_ratio
		self.avgpool_stride_ratio = avgpool_stride_ratio

		# how many features will be received after first convolution
		# value the same as in the original Torch code
		if self.bc_mode:
			self.first_output_features = growth_rate * 2
		else:
			self.first_output_features = growth_rate
		self.total_blocks = total_blocks
		self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks

		# compression rate at the transition layers
		self.reduction = reduction
		self.is_training = is_training

		self.initial_kernel = init_kernel
		self.comp_kernel = comp_kernel
		self.init_sigmas = init_sigmas
		self.comp_sigmas = comp_sigmas
		self.thetas = thetas
		self.phis = phis

		self.hermit_initial = init_basis_hermite_3D_steerable(self.initial_kernel, self.init_sigmas, theta=self.thetas[0], phi=self.phis[0], order=init_order) \
			if rfnn is not "single" else init_basis_hermite_3D(self.initial_kernel, self.init_sigmas[0], init_order)
		self.hermit_composite = init_basis_hermite_3D_steerable(self.comp_kernel, self.comp_sigmas, theta=self.thetas[1], phi=self.phis[1], order=comp_order) \
			if rfnn is not "single" else init_basis_hermite_3D(self.comp_kernel, self.comp_sigmas[0], comp_order)

		self.rfnn_layer = \
			_rfnn_conv_layer_pure_3d_SO_learn_sq_bc if rfnn=="learn_sq" else \
			_rfnn_conv_layer_pure_3d

		if not bc_mode:
			print("Build %s model with %d blocks, "
				  "%d composite layers each." % (
					  model_type, self.total_blocks, self.layers_per_block))
		if bc_mode:
			self.layers_per_block = self.layers_per_block // 2
			print("Build %s model with %d blocks, "
				  "%d bottleneck layers and %d composite layers each." % (
					  model_type, self.total_blocks, self.layers_per_block,
					  self.layers_per_block))
		print("Reduction at transition layers: %.1f" % self.reduction)

		self.keep_prob = keep_prob
		self.model_type = model_type

	def _count_trainable_params(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parametes = 1
			for dim in shape:
				variable_parametes *= dim.value
			total_parameters += variable_parametes
		if total_parameters / 1e6 < 1:
			print("Total training params: %.1fK" % (total_parameters / 1e3))
		else:
			print("Total training params: %.1fM" % (total_parameters / 1e6))

	def composite_function(self, _input, out_features, kernel_size=3):
		"""Function from paper H_l that performs:
		- batch normalization
		- ReLU nonlinearity
		- convolution with required kernel
		- dropout, if required
		"""
		with tf.variable_scope("composite_function"):
			# BN
#			output = self.batch_norm(_input)
			# ReLU
			output = tf.nn.relu(_input)
			# convolution
			if kernel_size == 1:
				output, weights = _conv_layer_pure_3d(output, shape=[1, 1, 1, output.get_shape()[-1].value, out_features], padding='VALID')
#				self.bc_weights.append(weights)
#				self.bc_conv_act.append(output)
			else:
				output, alphas, _ = self.rfnn_layer(output, self.hermit_composite, out_features)
#				self.kernels.append(kernel)
				self.alphas.append(alphas)
				self.conv_act.append(output)
			# dropout(in case of training and in case it is no 1.0)
			output = self.dropout(output)
		return output

	def bottleneck(self, _input, out_features):
		with tf.variable_scope("bottleneck"):
			# BN
#			output = self.batch_norm(_input)
			# ReLU
			output = tf.nn.relu(_input)
			inter_features = out_features * 2
			# 1x1 convolution
			output, weights = _conv_layer_pure_3d(output, shape=[1, 1, 1, output.get_shape()[-1].value, inter_features], padding='VALID')
#			self.bc_conv_act.append(output)
#			self.bc_weights.append(weights)
		output = self.dropout(output)
		return output

	def add_internal_layer(self, _input, growth_rate):
		"""Perform H_l composite function for the layer and after concatenate
		input with output from composite function.
		"""
		# call composite function with 3x3 kernel
		if not self.bc_mode:
			comp_out = self.composite_function(
				_input, out_features=growth_rate, kernel_size=self.comp_kernel)
		elif self.bc_mode:
			bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
			comp_out = self.composite_function(
				bottleneck_out, out_features=growth_rate, kernel_size=self.comp_kernel)
		# concatenate _input with out from composite function
		if TF_VERSION >= 1.0:
			output = tf.concat(axis=4, values=(_input, comp_out))
		else:
			output = tf.concat(4, (_input, comp_out))
		return output

	def add_block(self, _input, growth_rate, layers_per_block):
		"""Add N H_l internal layers"""
		output = _input
		for layer in range(layers_per_block):
			with tf.variable_scope("layer_%d" % layer):
				output = self.add_internal_layer(output, growth_rate)
		print(output.get_shape())
		return output

	def transition_layer(self, _input):
		"""Call H_l composite function with 1x1 kernel and after average
		pooling
		"""
		# call composite function with 1x1 kernel
		out_features = int(int(_input.get_shape()[-1]) * self.reduction)
		output = self.composite_function(
			_input, out_features=out_features, kernel_size=1)
		print(output.get_shape())
		# run average pooling
		output = self.avg_pool(output, k=2, s=2)
		print(output.get_shape())
		return output

	def transition_layer_to_classes(self, _input):
		"""This is last transition to get probabilities by classes. It perform:
		- batch normalization
		- ReLU nonlinearity
		- wide average pooling
		- FC layer multiplication
		"""
		# BN
#		output = self.batch_norm(_input)
		# ReLU
		output = tf.nn.relu(_input)
		# average pooling
		last_pool_kernel = [1, int(output.get_shape()[1].value), int(output.get_shape()[2]) * self.avgpool_kernel_ratio,
							int(output.get_shape()[3].value), 1]
		last_pool_stride = [1, int(output.get_shape()[1].value), int(output.get_shape()[2]) * self.avgpool_stride_ratio,
							int(output.get_shape()[3].value), 1]
		output = tf.cast(tf.nn.avg_pool3d(tf.cast(output, tf.float32), last_pool_kernel, last_pool_stride, 'VALID'), tf.float16)
		print(output.get_shape())
		# FC
		features_total = int(output.get_shape()[-1]) * int(output.get_shape()[-2]) * int(output.get_shape()[-3])
		output = tf.reshape(output, [-1, features_total])
		W = self.weight_variable_msra(
			[features_total, self.n_classes], name='W')
		self.weights.append(W)
		bias = self.bias_variable([self.n_classes])
		logits = tf.matmul(output, W) + bias
		self.fl_act.append(logits)

		return logits

	def avg_pool(self, _input, k, s):
		ksize = [1, k, k, k, 1]
		strides = [1, s, s, s, 1]
		padding = 'VALID'
		output = tf.cast(tf.nn.avg_pool3d(tf.cast(_input, tf.float32), ksize, strides, padding), tf.float16)
		return output

	def dropout(self, _input):
		if self.keep_prob < 1:
			output = tf.cond(
				self.is_training,
#				lambda: tf.cast(tf.nn.dropout(tf.cast(_input, dtype=tf.float16), self.keep_prob), dtype=tf.float32),
				lambda: tf.nn.dropout(_input, self.keep_prob),
				lambda: _input
			)
		else:
			output = _input
		return output

	def weight_variable_msra(self, shape, name):
		return tf.get_variable(
			name=name,
			shape=shape,
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float16)

	def weight_variable_xavier(self, shape, name):
		return tf.get_variable(
			name,
			shape=shape,
			initializer=tf.contrib.layers.xavier_initializer())

	def bias_variable(self, shape, name='bias'):
		initial = tf.constant(0.0, shape=shape, dtype=tf.float16)
		return tf.get_variable(name, initializer=initial, dtype=tf.float16)

	def inference(self, X):
		growth_rate = self.growth_rate
		layers_per_block = self.layers_per_block
		# first - initial 3 x 3 conv to first_output_features
		with tf.variable_scope("Initial_convolution"):
			output, alphas, kernel = self.rfnn_layer(X, self.hermit_initial, self.first_output_features, strides=[1,2,2,2,1])
			print(output.get_shape())
			self.kernels.append(kernel)
			self.alphas.append(alphas)
			self.conv_act.append(output)
		with tf.variable_scope("Initial_pooling"):
			output = tf.cast(tf.nn.max_pool3d(tf.cast(output, tf.float32), ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID'), tf.float16)
			print(output.get_shape())

		# add N required blocks
		for block in range(self.total_blocks):
			with tf.variable_scope("Block_%d" % block):
				output = self.add_block(output, growth_rate, layers_per_block)
			# last block exist without transition layer
			if block != self.total_blocks - 1:
				with tf.variable_scope("Transition_after_block_%d" % block):
					output = self.transition_layer(output)

		with tf.variable_scope("Transition_to_classes"):
			logits = self.transition_layer_to_classes(output)

		self._count_trainable_params()

		return logits
