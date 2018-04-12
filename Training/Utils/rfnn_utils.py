import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import convolve1d as conv
from Utils.cnn_utils import batch_norm_wrapper
from scipy.ndimage.interpolation import rotate


def _rfnn_conv_layer_2d_with_alphas(input, basis, alphas, biases, strides, padding, is_training, bnorm=False):
	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :])
			, axis=2, name='weights_' + str(i))

		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
		if bnorm:
			conv = batch_norm_wrapper(conv, is_training, True)
		else:
			conv = tf.nn.bias_add(conv, biases)
		conv_out = tf.nn.relu(conv, name='Activation')

		kernels.append(kernel)
		outputs.append(conv_out)

	return outputs, kernels

def _rfnn_conv_layer_2d(input, basis, omaps, strides, padding, is_training, bnorm=False):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[1]],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	if not bnorm:
		with tf.device('/cpu:0'):
			biases = tf.get_variable(
				'biases',
				shape=[omaps],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32)

	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :])
			, axis=2, name='weights_' + str(i))

		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
		if bnorm:
			conv = batch_norm_wrapper(conv, is_training, True)
		else:
			conv = tf.nn.bias_add(conv, biases)
		conv_out = tf.nn.relu(conv, name='Activation')

		kernels.append(kernel)
		outputs.append(conv_out)

		with tf.variable_scope('sigma%d' % i):
			# scale weights to [0 1], type is still float
			kernel_avg = tf.reduce_mean(kernel, axis=2)
			x_min = tf.reduce_min(kernel_avg)
			x_max = tf.reduce_max(kernel_avg)
			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)

			# to tf.image_summary format [batch_size, height, width, channels]
			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
			batch = kernel_transposed.get_shape()[0].value

			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return alphas, outputs, kernels


def _rfnn_conv_layer_pure_2d(input, basis, omaps, is_training, bn_mom, rn_mom, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)
	kernel = tf.reduce_sum(
		tf.transpose(alphas[:, :, :, None, None]) *
		tf.transpose(basis[None, None, :, :, :])
		, axis=2, name='weights')
	output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas, kernel

def _rfnn_conv_layer_pure_2d_fast(input, basis, omaps, init, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[input.get_shape()[-1].value, omaps, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-init, init),
			dtype=tf.float32)
	basis = tf.expand_dims(basis, 3)
#	basis = tf.tile(basis, [input.get_shape()[-1].value,1,1,1])
#	print(basis.get_shape())
	basis = tf.transpose(basis, (1,2,3,0))
#	print(basis.get_shape())
	conv_out = []
	for i in range(input.get_shape()[-1].value):
		conv_out.append(tf.nn.conv2d(input[:,:,:,i,None], basis, strides=strides, padding=padding))
	conv_out = tf.transpose(tf.stack(conv_out), (1,2,3,0,4))
#	print(conv_out.get_shape())
	output = tf.reduce_sum(
		alphas[None, None, None, :, :] *
		conv_out[:, :, :, :, None, :]
		, axis=-1)
	output = tf.reduce_sum(output, axis=-2)
#	print(output.get_shape())


	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas

def _rfnn_conv_layer_pure_2d_scales_max2(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-0.05, 0.05),
			dtype=tf.float32)

#	print(basis.get_shape())

	# Create convolutional kernel
	kernel = tf.reduce_sum(
		tf.transpose(alphas[:, :, :, None, None]) * tf.transpose(basis[None, None, :, :, :, :])
		, axis=3, name='weights')
#	print(kernel.get_shape())

	kernel = tf.transpose(kernel, [0, 1, 3, 2, 4])
#	print(kernel.get_shape())
	kernel_merged = tf.stack(kernel[:,:,:,:,0])
	for i in range(1, kernel.get_shape()[-1].value):
		kernel_merged = tf.concat(axis=-1, values=(kernel_merged, kernel[:,:,:,:,i]))
#	print(kernel_merged.get_shape())

	# Compute convolution
	outputs = tf.nn.conv2d(input, kernel_merged, strides=strides, padding=padding)
#	print(outputs.get_shape())

	# Max pool over scales
	scales = kernel.get_shape()[3].value
#	print(scales)

	output = tf.reduce_max(outputs[:,:,:,0:scales], reduction_indices=[-1], keep_dims=True)
	for i in range(1, kernel.get_shape()[-1].value):
#		print(output.get_shape())
		output = tf.concat(axis=-1, values=(output, tf.reduce_max(
				outputs[:,:,:,i*scales:(i+1)*scales], reduction_indices=[-1], keep_dims=True)))

#	print(output.get_shape())
	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas

def _rfnn_conv_layer_pure_2d_scales_max(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-0.05, 0.05),
			dtype=tf.float32)

	scales = np.shape(basis)[1]
	basis = tf.transpose(basis, [1, 0, 2, 3])
	kernels = []
	outputs = []
	for i in range(scales):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :])
			, axis=2, name='weights_' + str(i))

		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

		kernels.append(kernel)
		outputs.append(conv)

	outputs = tf.stack(outputs)
	output = tf.reduce_max(outputs, reduction_indices=[0])

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_scales_avg(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-0.05, 0.05),
			dtype=tf.float32)

	scales = np.shape(basis)[1]
	basis = tf.transpose(basis, [1, 0, 2, 3])
	kernels = []
	outputs = []
	for i in range(scales):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :])
			, axis=2, name='weights_' + str(i))

		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

		kernels.append(kernel)
		outputs.append(conv)

	outputs = tf.stack(outputs)
	output = tf.reduce_mean(outputs, reduction_indices=[0])

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_scales_learn(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-0.05, 0.05),
			dtype=tf.float32)

	kernel = tf.reduce_sum(
		tf.transpose(alphas[:, :, :, None, None]) *
		tf.transpose(basis[None, None, :, :, :, :])
		, axis=3, name='weights')

	merged_kernel = tf.reshape(kernel,
						shape=[kernel.get_shape()[0].value, kernel.get_shape()[1].value,
							   kernel.get_shape()[3].value,
							   kernel.get_shape()[2].value * kernel.get_shape()[4].value])
#	print(merged_kernel.get_shape())

	merged_outputs = tf.nn.conv2d(input, merged_kernel, strides=strides, padding=padding)
#	print(merged_outputs.get_shape())

	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, merged_kernel.get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	output = tf.nn.conv2d(merged_outputs, scale_bc, strides=[1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_scales_learn_bc(input, basis, omaps, is_training, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[0]],
			initializer=tf.random_uniform_initializer(-0.05, 0.05),
			dtype=tf.float32)

	kernel = tf.reduce_sum(
		tf.transpose(alphas[:, :, :, None, None]) *
		tf.transpose(basis[None, None, :, :, :, :])
		, axis=3, name='weights')

	merged_kernel = tf.reshape(kernel,
						shape=[kernel.get_shape()[0].value, kernel.get_shape()[1].value,
							   kernel.get_shape()[3].value,
							   kernel.get_shape()[2].value * kernel.get_shape()[4].value])
#	print(merged_kernel.get_shape())

	merged_outputs = tf.nn.conv2d(input, merged_kernel, strides=strides, padding=padding)
#	print(merged_outputs.get_shape())

	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, merged_kernel.get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	# BN
	output = batch_norm(merged_outputs, is_training=is_training)
	# ReLU
	output = tf.nn.relu(output)
	# Conv
	output = tf.nn.conv2d(output, scale_bc, [1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas

def _rfnn_conv_layer_pure_2d_scales_learn_flatten(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, basis.get_shape()[0].value*basis.get_shape()[1].value],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	basis = tf.reshape(basis, (basis.get_shape()[0].value*basis.get_shape()[1].value,
							   basis.get_shape()[2].value, basis.get_shape()[3].value))

	kernel = tf.reduce_sum(
		tf.transpose(alphas[:, :, :, None, None]) *
		tf.transpose(basis[None, None, :, :, :])
		, axis=2, name='weights')

	output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
	#	print(merged_outputs.get_shape())

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas


# === keeping output size intact ===
def _rfnn_conv_layer_pure_2d_SO_max(input, basis, omaps, is_training, bn_mom, rn_mom, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	scales = np.shape(basis)[0]
	orientations = np.shape(basis)[1]

	kernels = []
	outputs = []
	for i in range(scales):
		for j in range(orientations):
			kernel = tf.reduce_sum(
				alphas[None, None, :, None, None, :, :] * basis[i, j, :, :, :, None, None]
				, axis=2)

			kernel = kernel[0,0,:,:,:,:]
			conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

			kernels.append(kernel)
			outputs.append(conv)

	outputs = tf.stack(outputs)
#	print(outputs.get_shape())
	output = tf.reduce_max(outputs, reduction_indices=[0])
#	kernel = tf.reduce_max(kernels, reduction_indices=[0])

	kernels = tf.stack(kernels)
	kernels = tf.transpose(kernels, (1,2,3,4,0))
	kernels = tf.reshape(kernels, (kernels.get_shape()[0].value,kernels.get_shape()[1].value,
								   kernels.get_shape()[2].value,
								   kernels.get_shape()[3].value*kernels.get_shape()[4].value))

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_SO_avg(input, basis, omaps, is_training, bn_mom, rn_mom, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	scales = np.shape(basis)[0]
	orientations = np.shape(basis)[1]

	kernels = []
	outputs = []
	for i in range(scales):
		for j in range(orientations):
			kernel = tf.reduce_sum(
				alphas[None, None, :, None, None, :, :] * basis[i, j, :, :, :, None, None]
				, axis=2, name='weights_' + str(i))

			kernel = kernel[0,0,:,:,:,:]
			conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

			kernels.append(kernel)
			outputs.append(conv)

	outputs = tf.stack(outputs)
	output = tf.reduce_mean(outputs, reduction_indices=[0])
#	kernel = tf.reduce_mean(kernels, reduction_indices=[0])

	kernels = tf.stack(kernels)
	kernels = tf.transpose(kernels, (1,2,3,4,0))
	kernels = tf.reshape(kernels, (kernels.get_shape()[0].value,kernels.get_shape()[1].value,
								   kernels.get_shape()[2].value,
								   kernels.get_shape()[3].value*kernels.get_shape()[4].value))

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_SO_learn_sq_bc(input, basis, omaps, is_training, bn_mom, rn_mom, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	scales = np.shape(basis)[0]
	orientations = np.shape(basis)[1]

	outputs = []
	kernels = []
	for i in range(scales):
		for j in range(orientations):
			kernel = tf.reduce_sum(
				alphas[None, None, :, None, None, :, :] * basis[i, j, :, :, :, None, None]
				, axis=2, name='weights_' + str(i) + str(j))

			kernel = kernel[0,0,:,:,:,:]
			kernels.append(kernel)

			conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

			outputs.append(conv)

	outputs = tf.stack(outputs)
	outputs = tf.transpose(outputs, (1,2,3,0,4))
	outputs = tf.reshape(outputs, (outputs.get_shape()[0].value, outputs.get_shape()[1].value, outputs.get_shape()[2].value,
								   outputs.get_shape()[3].value * outputs.get_shape()[4].value))
	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, outputs.get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	# BN
	outputs = batch_norm(outputs, is_training, bn_mom, rn_mom)
	# ReLU
	outputs = tf.nn.relu(outputs)
	# Conv
	output = tf.nn.conv2d(outputs, scale_bc, [1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	kernels = tf.stack(kernels)
	kernels = tf.transpose(kernels, (1,2,3,4,0))
	kernels = tf.reshape(kernels, (kernels.get_shape()[0].value,kernels.get_shape()[1].value,
								   kernels.get_shape()[2].value,
								   kernels.get_shape()[3].value*kernels.get_shape()[4].value))

	return output, alphas, kernels

def _rfnn_conv_layer_pure_2d_SO_learn_fl_bc(input, basis, omaps, strides=[1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	kernel = tf.reduce_sum(alphas[None,None,:,None,None,:,:] * basis[:,:,:,:,:,None,None]
		,axis=2, name='weights')

	# to shape: [k,k,i,scales,orientations,o]
	kernel = tf.transpose(kernel, (2,3,4,0,1,5))
	kernel = tf.reshape(kernel, (kernel.get_shape()[0].value, kernel.get_shape()[1].value, kernel.get_shape()[2].value,
								 kernel.get_shape()[3].value*kernel.get_shape()[4].value*kernel.get_shape()[5].value))

	output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, output.get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	# BN
#	output = batch_norm(output, is_training=is_training)
	# ReLU
#	output = tf.nn.relu(output)
	# Conv
	output = tf.nn.conv2d(output, scale_bc, [1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas, kernel

# === altered output size, omaps*S*O ===
def _rfnn_conv_layer_pure_2d_SO_learn_flatten(input, basis, omaps, is_training, bn_mom, rn_mom, strides=[1, 1, 1, 1], padding='SAME'):
	# flattening all scales and orientation -> output size = omaps*scales*orientations
	# number of alphas stays omaps
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	kernel = tf.reduce_sum(alphas[None,None,:,None,None,:,:] * basis[:,:,:,:,:,None,None]
		,axis=2, name='weights')

	# to shape: [k,k,i,scales,orientations,o]
	kernel = tf.transpose(kernel, (2,3,4,0,1,5))
	kernel = tf.reshape(kernel, (kernel.get_shape()[0].value, kernel.get_shape()[1].value, kernel.get_shape()[2].value,
								 kernel.get_shape()[3].value*kernel.get_shape()[4].value*kernel.get_shape()[5].value))

	output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
#	print(output.get_shape())
	#	print(merged_outputs.get_shape())

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas, kernel



def _rfnn_deconv_layer_2d(input, basis, omaps, oshape, strides, padding, bnorm=False):
	alphas = tf.get_variable(
		'alphas',
		shape=[input.get_shape()[-1].value, omaps, np.shape(basis)[1]],
		initializer=tf.random_uniform_initializer(-1.0, 1.0),
		dtype=tf.float32)

	biases = tf.get_variable(
		'biases',
		shape=[omaps],
		initializer=tf.constant_initializer(0.0),
		dtype=tf.float32)

	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :])
			, axis=2, name='weights_' + str(i))

		deconv = tf.nn.conv2d_transpose(input, kernel, oshape, strides, padding)
		deconv = tf.nn.bias_add(deconv, biases)
		deconv = tf.nn.relu(deconv, name='Activation')

		outputs.append(deconv)

	return outputs

# ======== 3D layers ==========

def _rfnn_conv_layer_3d(input, basis, omaps, strides, padding, is_training, bnorm=False):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[1]],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	if not bnorm:
		with tf.device('/cpu:0'):
			biases = tf.get_variable(
				'biases',
				shape=[omaps],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32)

	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :, :])
			, axis=3, name='weights_' + str(i))

		conv = tf.cast(
			tf.nn.conv3d(tf.cast(input, tf.float32), tf.cast(kernel, tf.float32), strides=strides, padding=padding),
			tf.float32)
		if bnorm:
			conv = batch_norm_wrapper(conv, is_training, True)
		else:
			conv = tf.nn.bias_add(conv, biases)
		conv_out = tf.nn.relu(conv, name='Activation')

		kernels.append(kernel)
		outputs.append(conv_out)

	return alphas, outputs, kernels

def _rfnn_conv_layer_pure_3d(input, basis, omaps, strides=[1, 1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[np.shape(basis)[0], input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	kernel = tf.reduce_sum(
		alphas[:, None, None, None, :, :] * basis[:, :, :, :, None, None]
		, axis=0, name='weights')

	output = tf.nn.conv3d(input, kernel, strides=strides, padding=padding)

	return output, alphas, kernel

def _rfnn_conv_layer_pure_3d_SO_learn_sq_bc(input, basis, omaps, strides=[1, 1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	scales = np.shape(basis)[0]
	orientations = np.shape(basis)[1]

	outputs = []
	kernels = []
	for i in range(scales):
		for j in range(orientations):
			kernel = tf.reduce_sum(
				alphas[None, None, :, None, None, None, :, :] * basis[i, j, :, :, :, :, None, None]
				, axis=2, name='weights_' + str(i) + str(j))

			kernel = kernel[0,0,:,:,:,:]
			kernels.append(kernel)

			conv = tf.nn.conv3d(input, kernel, strides=strides, padding=padding)

			outputs.append(conv)

	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, 1, len(outputs)*outputs[0].get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	# BN
#	outputs = batch_norm(outputs, is_training, bn_mom, rn_mom)
	# ReLU
	outputs = tf.nn.relu(
		tf.reshape(
			tf.transpose(
				tf.stack(outputs), (1,2,3,4,0,5)),
				(outputs[0].get_shape()[0].value, outputs[0].get_shape()[1].value, outputs[0].get_shape()[2].value,
						outputs[0].get_shape()[3].value, len(outputs) * outputs[0].get_shape()[4].value)))
	# Conv
	output = tf.nn.conv3d(outputs, scale_bc, [1, 1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	kernels = tf.stack(kernels)
	kernels = tf.transpose(kernels, (1,2,3,4,5,0))
	kernels = tf.reshape(kernels, (kernels.get_shape()[0].value,kernels.get_shape()[1].value,kernels.get_shape()[2].value,
								   kernels.get_shape()[3].value,
								   kernels.get_shape()[4].value*kernels.get_shape()[5].value))

	return output, alphas, kernels

def _rfnn_conv_layer_pure_3d_SO_learn_fl_bc(input, basis, omaps, strides=[1, 1, 1, 1, 1], padding='SAME'):
	with tf.device('/cpu:0'):
		alphas = tf.get_variable(
			'alphas',
			shape=[basis.get_shape()[2].value, input.get_shape()[-1].value, omaps],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

	kernel = tf.reduce_sum(alphas[None,None,:,None,None,None,:,:] * basis[:,:,:,:,:,:,None,None]
		,axis=2, name='weights')

	# to shape: [k,k,i,scales,orientations,o]
	kernel = tf.transpose(kernel, (2,3,4,5,0,1,6))
	kernel = tf.reshape(kernel, (kernel.get_shape()[0].value, kernel.get_shape()[1].value, kernel.get_shape()[2].value,
	 								kernel.get_shape()[3].value,
								 kernel.get_shape()[4].value*kernel.get_shape()[5].value*kernel.get_shape()[6].value))

	output = tf.nn.conv3d(input, kernel, strides=strides, padding=padding)

	with tf.device('/cpu:0'):
		scale_bc = tf.get_variable(
			'scale_bottleneck',
			shape=[1, 1, 1, output.get_shape()[-1].value, omaps],
			initializer=tf.contrib.layers.variance_scaling_initializer(),
			dtype=tf.float32)

	# BN
#	output = batch_norm(output, is_training=is_training)
	# ReLU
#	output = tf.nn.relu(output)
	# Conv
	output = tf.nn.conv3d(output, scale_bc, [1, 1, 1, 1, 1], padding='SAME')

	#		with tf.variable_scope('sigma%d' % i):
	#			# scale weights to [0 1], type is still float
	#			kernel_avg = tf.reduce_mean(kernel, axis=2)
	#			x_min = tf.reduce_min(kernel_avg)
	#			x_max = tf.reduce_max(kernel_avg)
	#			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
	#
	#			# to tf.image_summary format [batch_size, height, width, channels]
	#			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
	#			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
	#			batch = kernel_transposed.get_shape()[0].value
	#
	#			tf.summary.image('filters', kernel_transposed, max_outputs=batch)

	return output, alphas, kernel

def _rfnn_deconv_layer_3d(input, basis, omaps, oshape, strides, padding, bnorm=False):
	alphas = tf.get_variable(
		'alphas',
		shape=[input.get_shape()[-1].value, omaps, np.shape(basis)[1]],
		initializer=tf.random_uniform_initializer(-1.0, 1.0),
		dtype=tf.float32)

	biases = tf.get_variable(
		'biases',
		shape=[omaps],
		initializer=tf.constant_initializer(0.0),
		dtype=tf.float32)

	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
			tf.transpose(alphas[:, :, :, None, None, None]) *
			tf.transpose(basis[None, None, i, :, :, :, :])
			, axis=3, name='weights_' + str(i))

		deconv = tf.nn.conv3d_transpose(input, kernel, oshape, strides, padding)
		deconv = tf.nn.bias_add(deconv, biases)
		deconv = tf.nn.relu(deconv, name='Activation')

		outputs.append(deconv)

	return outputs


def init_basis_hermite_2D(kernel, sigma, order):
	nrBasis = 15
	hermiteBasis = np.empty((np.int(nrBasis), np.int(kernel), np.int(kernel)))
	threshold = 3 if order == 1 \
		else 6 if order == 2 \
		else 10 if order == 3 \
		else 15

	x = np.arange(-np.int(kernel - 1 / 2), np.int(kernel - 1 / 2) + 1, dtype=np.float32)
	impulse = np.zeros((kernel, kernel))
	impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

	g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
	#    g = g / g.sum()
	g1 = sigma * -(x / np.square(sigma)) * g
	g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
	g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
	g4 = np.power(sigma, 4) * (
		((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g
	gauss0x = conv(impulse, g, axis=1)
	gauss0y = conv(impulse, g, axis=0)
	gauss1x = conv(impulse, g1, axis=1)
	gauss1y = conv(impulse, g1, axis=0)
	gauss2x = conv(impulse, g2, axis=1)
	gauss0 = conv(gauss0x, g, axis=0)

	hermiteBasis[0, :, :] = gauss0  # g
	hermiteBasis[1, :, :] = conv(gauss0y, g1, axis=1)  # g_x
	hermiteBasis[2, :, :] = conv(gauss0x, g1, axis=0)  # g_y
	hermiteBasis[3, :, :] = conv(gauss0y, g2, axis=1)  # g_xx
	hermiteBasis[4, :, :] = conv(gauss0x, g2, axis=0)  # g_yy
	hermiteBasis[5, :, :] = conv(gauss1x, g1, axis=0)  # g_xy
	hermiteBasis[6, :, :] = conv(gauss0y, g3, axis=1)  # g_xxx
	hermiteBasis[7, :, :] = conv(gauss0x, g3, axis=0)  # g_yyy
	hermiteBasis[8, :, :] = conv(gauss1y, g2, axis=1)  # g_xxy
	hermiteBasis[9, :, :] = conv(gauss1x, g2, axis=0)  # g_yyx
	hermiteBasis[10, :, :] = conv(gauss0y, g4, axis=1)  # g_xxxx
	hermiteBasis[11, :, :] = conv(gauss0x, g4, axis=0)  # g_yyyy
	hermiteBasis[12, :, :] = conv(gauss1y, g3, axis=1)  # g_xxxy
	hermiteBasis[13, :, :] = conv(gauss1x, g3, axis=0)  # g_yyyx
	hermiteBasis[14, :, :] = conv(gauss2x, g2, axis=0)  # g_yyxx

#	for i in range(nrBasis):
#		hermiteBasis[i, :, :] /= rms(hermiteBasis[i, :, :])

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[0:threshold, :, :], dtype=tf.float32)

def init_basis_hermite_2D_scales(kernel, sigmas, order):
	nrBasis = 15
	hermiteBasis = np.empty((np.int(nrBasis), np.int(np.shape(sigmas)[0]), np.int(kernel), np.int(kernel)))
	threshold = 3 if order == 1 \
		else 6 if order == 2 \
		else 10 if order == 3 \
		else 15

	for i, sigma in enumerate(sigmas):
		x = np.arange(-np.int(kernel - 1 / 2), np.int(kernel - 1 / 2) + 1, dtype=np.float32)
		impulse = np.zeros((kernel, kernel))
		impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

		g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
		g = g / g.sum()
		g1 = sigma * -(x / np.square(sigma)) * g
		g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
		g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
		g4 = np.power(sigma, 4) * (
			((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g
		gauss0x = conv(impulse, g, axis=1)
		gauss0y = conv(impulse, g, axis=0)
		gauss1x = conv(impulse, g1, axis=1)
		gauss1y = conv(impulse, g1, axis=0)
		gauss2x = conv(impulse, g2, axis=1)
		gauss0 = conv(gauss0x, g, axis=0)

		hermiteBasis[0, i, :, :] = gauss0					  # g
		hermiteBasis[1, i, :, :] = conv(gauss0y, g1, axis=1)  # g_x
		hermiteBasis[2, i, :, :] = conv(gauss0x, g1, axis=0)  # g_y
		hermiteBasis[3, i, :, :] = conv(gauss0y, g2, axis=1)  # g_xx
		hermiteBasis[4, i, :, :] = conv(gauss0x, g2, axis=0)  # g_yy
		hermiteBasis[5, i, :, :] = conv(gauss1x, g1, axis=0)  # g_xy
		hermiteBasis[6, i, :, :] = conv(gauss0y, g3, axis=1)  # g_xxx
		hermiteBasis[7, i, :, :] = conv(gauss0x, g3, axis=0)  # g_yyy
		hermiteBasis[8, i, :, :] = conv(gauss1y, g2, axis=1)  # g_xxy
		hermiteBasis[9, i, :, :] = conv(gauss1x, g2, axis=0)  # g_yyx
		hermiteBasis[10, i, :, :] = conv(gauss0y, g4, axis=1)  # g_xxxx
		hermiteBasis[11, i, :, :] = conv(gauss0x, g4, axis=0)  # g_yyyy
		hermiteBasis[12, i, :, :] = conv(gauss1y, g3, axis=1)  # g_xxxy
		hermiteBasis[13, i, :, :] = conv(gauss1x, g3, axis=0)  # g_yyyx
		hermiteBasis[14, i, :, :] = conv(gauss2x, g2, axis=0)  # g_yyxx

#	for i, sigma in enumerate(sigmas):
#		for j in range(nrBasis):
#			hermiteBasis[j, i, :, :] /= rms(hermiteBasis[j, i, :, :])

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[0:threshold, :, :, :], dtype=tf.float32)

def init_basis_hermite_steerable_2D(kernel, sigmas, theta=90.0, order=4):
	angle = theta * np.pi / 180.0
	orients1 = np.int(180 / theta)
	hermiteBasis = np.zeros(
		(np.int(np.shape(sigmas)[0]), orients1, 5, np.int(kernel), np.int(kernel)))

	for i, sigma in enumerate(sigmas):
		x = np.arange(-(np.int(kernel - 1)) / 2, (np.int(kernel - 1)) / 2 + 1, dtype=np.float32)
		impulse = np.zeros((np.int(kernel), np.int(kernel)))
		impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

		g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
		g1 = sigma * -(x / np.square(sigma)) * g
		g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
		g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
		g4 = np.power(sigma, 4) * (
			((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g
		gauss0x = conv(impulse, g, axis=1)
		gauss0y = conv(impulse, g, axis=0)
		gauss0 = conv(gauss0x, g, axis=0)

		for t in range(orients1):
			# g
			hermiteBasis[i, t, 0, :, :] = gauss0

			# g_y g_x
			hermiteBasis[i, t, 1, :, :] = np.cos(t * angle) * conv(gauss0y, g1, axis=1) \
										  + np.cos(np.pi / 2.0 - t * angle) * conv(gauss0x, g1, axis=0)

			# g_xx g_xy g_yy
			hermiteBasis[i, t, 2, :, :] = \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(gauss0x, g2, axis=0) \
				+ np.power(np.cos(t * angle), 2) * conv(gauss0y, g2, axis=1) \
				+ 2 * np.cos(t * angle) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g1, axis=0), g1, axis=1)

			# g_xxx g_xxy g_yyx g_yyy
			hermiteBasis[i, t, 3, :, :] = \
				+ 3 * np.power(np.cos(t * angle), 2) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g2, axis=1),
																							  g1, axis=0) \
				+ 3 * np.cos(t * angle) * np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(conv(impulse, g2, axis=0),
																							  g1, axis=1) \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 3) * conv(gauss0x, g3, axis=0) \
				+ np.power(np.cos(t * angle), 3) * conv(gauss0y, g3, axis=1)

			# g_xxxx g_xxxy g_yyxx g_yyyx g_yyyy
			hermiteBasis[i, t, 4, :, :] = \
				np.power(np.cos(t * angle), 4) * conv(gauss0y, g4, axis=1) \
				+ 4 * np.power(np.cos(t * angle), 3) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g3, axis=1),
																							  g1, axis=0) \
				+ 6 * np.power(np.cos(t * angle), 2) * np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(
					conv(impulse, g2, axis=1), g2, axis=0) \
				+ 4 * np.cos(t * angle) * np.power(np.cos(np.pi / 2.0 - t * angle), 3) * conv(conv(impulse, g3, axis=0),
																							  g1, axis=1) \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 4) * conv(gauss0x, g4, axis=0)

		hermiteBasis[i, :, :, :, :] /= np.sqrt(sigma)

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[:, :, 0:order + 1, :, :], dtype=tf.float32)

def init_basis_hermite_steerable_full(kernel, sigmas, theta=90.0, order=4):
	angle = theta * np.pi / 180.0
	orients1 = np.int(180 / theta)
	threshold = 2 if order == 1 \
		else 4 if order == 2 \
		else 6 if order == 3 \
		else 9
	hermiteBasis = np.zeros(
		(np.int(np.shape(sigmas)[0]), orients1, 10, np.int(kernel), np.int(kernel)))
	for i, sigma in enumerate(sigmas):
		x = np.arange(-(np.int(kernel - 1)) / 2, (np.int(kernel - 1)) / 2 + 1, dtype=np.float32)
		impulse = np.zeros((np.int(kernel), np.int(kernel)))
		impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

		g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
		g1 = sigma * -(x / np.square(sigma)) * g
		g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
		g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
		g4 = np.power(sigma, 4) * (
			((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g
		gauss0x = conv(impulse, g, axis=1)
		gauss0y = conv(impulse, g, axis=0)
		gauss0 = conv(gauss0x, g, axis=0)

		for t in range(orients1):
			# g
			hermiteBasis[i, t, 0, :, :] = gauss0

			# g_y g_x
			hermiteBasis[i, t, 1, :, :] = np.cos(t * angle) * conv(gauss0y, g1, axis=1) \
										  + np.cos(np.pi / 2.0 - t * angle) * conv(gauss0x, g1, axis=0)

			# g_xx g_xy g_yy
			hermiteBasis[i, t, 2, :, :] = \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(gauss0x, g2, axis=0) \
				+ np.power(np.cos(t * angle), 2) * conv(gauss0y, g2, axis=1) \
				+ 2 * np.cos(t * angle) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g1, axis=0), g1, axis=1)

			hermiteBasis[i, t, 3, :, :] = \
						rotate(conv(conv(impulse, g1, axis=0), g1, axis=1), -t * theta, reshape=False)

			# g_xxx g_xxy g_yyx g_yyy
			hermiteBasis[i, t, 4, :, :] = \
				+ 3 * np.power(np.cos(t * angle), 2) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g2, axis=1),
																							  g1, axis=0) \
				+ 3 * np.cos(t * angle) * np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(conv(impulse, g2, axis=0),
																							  g1, axis=1) \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 3) * conv(gauss0x, g3, axis=0) \
				+ np.power(np.cos(t * angle), 3) * conv(gauss0y, g3, axis=1)

			hermiteBasis[i, t, 5, :, :] = \
				rotate(conv(conv(impulse, g2, axis=1), g1, axis=0), -t * theta, reshape=False)

			# g_xxxx g_xxxy g_yyxx g_yyyx g_yyyy
			hermiteBasis[i, t, 6, :, :] = \
				np.power(np.cos(t * angle), 4) * conv(gauss0y, g4, axis=1) \
				+ 4 * np.power(np.cos(t * angle), 3) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g3, axis=1),
																							  g1, axis=0) \
				+ 12 * np.power(np.cos(t * angle), 2) * np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(
					conv(impulse, g2, axis=1), g2, axis=0) \
				+ 4 * np.cos(t * angle) * np.power(np.cos(np.pi / 2.0 - t * angle), 3) * conv(conv(impulse, g3, axis=0),
																							  g1, axis=1) \
				+ np.power(np.cos(np.pi / 2.0 - t * angle), 4) * conv(gauss0x, g4, axis=0)

			hermiteBasis[i, t, 7, :, :] = \
				rotate(conv(conv(impulse, g3, axis=1), g1, axis=0), -t * theta, reshape=False)

			hermiteBasis[i, t, 8, :, :] = \
				rotate(conv(conv(impulse, g2, axis=1), g2, axis=0), -t * theta, reshape=False)

		hermiteBasis[i, :, :, :, :] /= np.sqrt(sigma)

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[:, :, 0:threshold, :, :], dtype=tf.float32)


def init_basis_hermite_3D(kernel, sigma, order):
	nrBasis = 35
	hermiteBasis = np.zeros(
		(np.int(nrBasis), np.int(kernel), np.int(kernel), np.int(kernel)))

	threshold = 4 if order == 1 \
		else 10 if order == 2 \
		else 20 if order == 3 \
		else 35

	x = np.arange(-np.int((kernel - 1) / 2), np.int((kernel - 1) / 2) + 1, dtype=np.float32)
	impulse = np.zeros((kernel, kernel, kernel))
	impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0


	g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
	g = g / g.sum()
	g1 = sigma * -(x / np.square(sigma)) * g
	g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
	g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
	g4 = np.power(sigma, 4) * (
		((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g

	# 0. order: g
	gauss0 = conv(conv(conv(impulse, g, axis=0), g, axis=1), g, axis=2)

	# 1. order: g_z, g_x, g_y
	gauss1z_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g1, axis=2)
	gauss1x_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g1, axis=1)
	gauss1y_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g1, axis=0)

	# 2. order: g_xx, g_yy, g_zz,
	#		   g_xy, g_xz, g_zy
	gauss2xx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g2, axis=1)
	gauss2yy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g2, axis=0)
	gauss2zz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g2, axis=2)

	gauss2xy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g1, axis=0)
	gauss2xz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g1, axis=2)
	gauss2zy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g1, axis=0)

	# 3. order: g_xxx, g_yyy, g_zzz
	#		   g_xxy, g_xxz, g_xyy, g_xzz, g_yyz, g_yzz, g_xyz
	gauss3xxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g3, axis=1)
	gauss3yyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g3, axis=0)
	gauss3zzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g3, axis=2)

	gauss3xxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g2, axis=1)
	gauss3xxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g2, axis=1)
	gauss3xyy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g2, axis=1)
	gauss3xzz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g2, axis=2)
	gauss3yyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g2, axis=0)
	gauss3yzz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g2, axis=2)

	gauss3xyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g1, axis=2)

	# 4. order: g_xxxx, g_yyyy, g_zzzz,
	#		   g_xxxy, g_xxxz, g_yyyx, g_yyyz, g_zzzx, g_zzzy,
	#		   g_xxyy, g_xxzz, g_xxyz, g_xyyz, g_yyzz, g_xzzy,
	gauss4xxxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g4, axis=1)
	gauss4yyyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g4, axis=0)
	gauss4zzzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g4, axis=2)

	gauss4xxxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g3, axis=1)
	gauss4xxxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g3, axis=1)
	gauss4yyyx_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g3, axis=0)
	gauss4yyyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g3, axis=0)
	gauss4zzzx_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g3, axis=2)
	gauss4zzzy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g3, axis=2)

	gauss4xxyy_ = conv(conv(conv(impulse, g, axis=2), g2, axis=0), g2, axis=1)
	gauss4xxzz_ = conv(conv(conv(impulse, g, axis=0), g2, axis=2), g2, axis=1)
	gauss4xxyz_ = conv(conv(conv(impulse, g1, axis=0), g1, axis=2), g2, axis=1)
	gauss4xyyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=2), g2, axis=0)
	gauss4yyzz_ = conv(conv(conv(impulse, g, axis=1), g2, axis=0), g2, axis=2)
	gauss4xzzy_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g2, axis=2)

	hermiteBasis[0, :, :, :] = gauss0
	hermiteBasis[1, :, :, :] = gauss1z_
	hermiteBasis[2, :, :, :] = gauss1x_
	hermiteBasis[3, :, :, :] = gauss1y_
	hermiteBasis[4, :, :, :] = gauss2xx_
	hermiteBasis[5, :, :, :] = gauss2yy_
	hermiteBasis[6, :, :, :] = gauss2zz_
	hermiteBasis[7, :, :, :] = gauss2xy_
	hermiteBasis[8, :, :, :] = gauss2xz_
	hermiteBasis[9, :, :, :] = gauss2zy_
	hermiteBasis[10, :, :, :] = gauss3xxx_
	hermiteBasis[11, :, :, :] = gauss3yyy_
	hermiteBasis[12, :, :, :] = gauss3zzz_
	hermiteBasis[13, :, :, :] = gauss3xxy_
	hermiteBasis[14, :, :, :] = gauss3xxz_
	hermiteBasis[15, :, :, :] = gauss3xyy_
	hermiteBasis[16, :, :, :] = gauss3xzz_
	hermiteBasis[17, :, :, :] = gauss3yyz_
	hermiteBasis[18, :, :, :] = gauss3yzz_
	hermiteBasis[19, :, :, :] = gauss3xyz_
	hermiteBasis[20, :, :, :] = gauss4xxxx_
	hermiteBasis[21, :, :, :] = gauss4yyyy_
	hermiteBasis[22, :, :, :] = gauss4zzzz_
	hermiteBasis[23, :, :, :] = gauss4xxxy_
	hermiteBasis[24, :, :, :] = gauss4xxxz_
	hermiteBasis[25, :, :, :] = gauss4yyyx_
	hermiteBasis[26, :, :, :] = gauss4yyyz_
	hermiteBasis[27, :, :, :] = gauss4zzzx_
	hermiteBasis[28, :, :, :] = gauss4zzzy_
	hermiteBasis[29, :, :, :] = gauss4xxyy_
	hermiteBasis[30, :, :, :] = gauss4xxzz_
	hermiteBasis[31, :, :, :] = gauss4xxyz_
	hermiteBasis[32, :, :, :] = gauss4xyyz_
	hermiteBasis[33, :, :, :] = gauss4yyzz_
	hermiteBasis[34, :, :, :] = gauss4xzzy_

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[0:threshold, :, :, :], dtype=tf.float32)

def init_basis_hermite_3D_scales(kernel, sigmas, order):
	nrBasis = 35
	hermiteBasis = np.empty(
		(np.int(np.shape(sigmas)[0]), np.int(nrBasis), np.int(kernel), np.int(kernel), np.int(kernel)))

	threshold = 4 if order == 1 \
		else 10 if order == 2 \
		else 20 if order == 3 \
		else 35

	x = np.arange(-np.int((kernel - 1) / 2), np.int((kernel - 1) / 2) + 1, dtype=np.float32)
	impulse = np.zeros((kernel, kernel, kernel))
	impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

	for i, sigma in enumerate(sigmas):
		g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
		g = g / g.sum()
		g1 = sigma * -(x / np.square(sigma)) * g
		g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
		g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
		g4 = np.power(sigma, 4) * (
			((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g

		# 0. order: g
		gauss0 = conv(conv(conv(impulse, g, axis=0), g, axis=1), g, axis=2)

		# 1. order: g_z, g_x, g_y
		gauss1z_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g1, axis=2)
		gauss1x_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g1, axis=1)
		gauss1y_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g1, axis=0)

		# 2. order: g_xx, g_yy, g_zz,
		#		   g_xy, g_xz, g_zy
		gauss2xx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g2, axis=1)
		gauss2yy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g2, axis=0)
		gauss2zz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g2, axis=2)

		gauss2xy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g1, axis=0)
		gauss2xz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g1, axis=2)
		gauss2zy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g1, axis=0)

		# 3. order: g_xxx, g_yyy, g_zzz
		#		   g_xxy, g_xxz, g_xyy, g_xzz, g_yyz, g_yzz, g_xyz
		gauss3xxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g3, axis=1)
		gauss3yyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g3, axis=0)
		gauss3zzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g3, axis=2)

		gauss3xxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g2, axis=1)
		gauss3xxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g2, axis=1)
		gauss3xyy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g2, axis=1)
		gauss3xzz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g2, axis=2)
		gauss3yyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g2, axis=0)
		gauss3yzz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g2, axis=2)

		gauss3xyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g1, axis=2)

		# 4. order: g_xxxx, g_yyyy, g_zzzz,
		#		   g_xxxy, g_xxxz, g_yyyx, g_yyyz, g_zzzx, g_zzzy,
		#		   g_xxyy, g_xxzz, g_xxyz, g_xyyz, g_yyzz, g_xzzy,
		gauss4xxxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g4, axis=1)
		gauss4yyyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g4, axis=0)
		gauss4zzzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g4, axis=2)

		gauss4xxxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g3, axis=1)
		gauss4xxxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g3, axis=1)
		gauss4yyyx_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g3, axis=0)
		gauss4yyyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g3, axis=0)
		gauss4zzzx_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g3, axis=2)
		gauss4zzzy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g3, axis=2)

		gauss4xxyy_ = conv(conv(conv(impulse, g, axis=2), g2, axis=0), g2, axis=1)
		gauss4xxzz_ = conv(conv(conv(impulse, g, axis=0), g2, axis=2), g2, axis=1)
		gauss4xxyz_ = conv(conv(conv(impulse, g1, axis=0), g1, axis=2), g2, axis=1)
		gauss4xyyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=2), g2, axis=0)
		gauss4yyzz_ = conv(conv(conv(impulse, g, axis=1), g2, axis=0), g2, axis=2)
		gauss4xzzy_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g2, axis=2)

		hermiteBasis[i, 0, :, :, :] = gauss0
		hermiteBasis[i, 1, :, :, :] = gauss1z_
		hermiteBasis[i, 2, :, :, :] = gauss1x_
		hermiteBasis[i, 3, :, :, :] = gauss1y_
		hermiteBasis[i, 4, :, :, :] = gauss2xx_
		hermiteBasis[i, 5, :, :, :] = gauss2yy_
		hermiteBasis[i, 6, :, :, :] = gauss2zz_
		hermiteBasis[i, 7, :, :, :] = gauss2xy_
		hermiteBasis[i, 8, :, :, :] = gauss2xz_
		hermiteBasis[i, 9, :, :, :] = gauss2zy_
		hermiteBasis[i, 10, :, :, :] = gauss3xxx_
		hermiteBasis[i, 11, :, :, :] = gauss3yyy_
		hermiteBasis[i, 12, :, :, :] = gauss3zzz_
		hermiteBasis[i, 13, :, :, :] = gauss3xxy_
		hermiteBasis[i, 14, :, :, :] = gauss3xxz_
		hermiteBasis[i, 15, :, :, :] = gauss3xyy_
		hermiteBasis[i, 16, :, :, :] = gauss3xzz_
		hermiteBasis[i, 17, :, :, :] = gauss3yyz_
		hermiteBasis[i, 18, :, :, :] = gauss3yzz_
		hermiteBasis[i, 19, :, :, :] = gauss3xyz_
		hermiteBasis[i, 20, :, :, :] = gauss4xxxx_
		hermiteBasis[i, 21, :, :, :] = gauss4yyyy_
		hermiteBasis[i, 22, :, :, :] = gauss4zzzz_
		hermiteBasis[i, 23, :, :, :] = gauss4xxxy_
		hermiteBasis[i, 24, :, :, :] = gauss4xxxz_
		hermiteBasis[i, 25, :, :, :] = gauss4yyyx_
		hermiteBasis[i, 26, :, :, :] = gauss4yyyz_
		hermiteBasis[i, 27, :, :, :] = gauss4zzzx_
		hermiteBasis[i, 28, :, :, :] = gauss4zzzy_
		hermiteBasis[i, 29, :, :, :] = gauss4xxyy_
		hermiteBasis[i, 30, :, :, :] = gauss4xxzz_
		hermiteBasis[i, 31, :, :, :] = gauss4xxyz_
		hermiteBasis[i, 32, :, :, :] = gauss4xyyz_
		hermiteBasis[i, 33, :, :, :] = gauss4yyzz_
		hermiteBasis[i, 34, :, :, :] = gauss4xzzy_

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[:, 0:threshold, :, :, :], dtype=tf.float32)

def init_basis_hermite_3D_steerable(kernel, sigmas, theta, phi, order=4):
	angle1 = theta * np.pi / 180
	angle2 = phi * np.pi / 180
	orients1 = np.int(180 / theta)
	orients2 = np.int(180 / phi)
	hermiteBasis = np.zeros(
		(np.int(np.shape(sigmas)[0]), orients1 * orients2, 5, np.int(kernel), np.int(kernel), np.int(kernel)))
	x = np.arange(-np.int((kernel - 1) / 2), np.int((kernel - 1) / 2) + 1, dtype=np.float32)
	impulse = np.zeros((kernel, kernel, kernel))
	impulse[np.int((kernel - 1) / 2), np.int((kernel - 1) / 2), np.int((kernel - 1) / 2)] = 1.0

	for i, sigma in enumerate(sigmas):
		g = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(np.square(x) / (-2 * np.square(sigma)))
		g = g / g.sum()
		g1 = sigma * -(x / np.square(sigma)) * g
		g2 = np.square(sigma) * ((np.square(x) - np.power(sigma, 2)) / np.power(sigma, 4)) * g
		g3 = np.power(sigma, 3) * -((np.power(x, 3) - 3 * x * np.square(sigma)) / np.power(sigma, 6)) * g
		g4 = np.power(sigma, 4) * (
		((np.power(x, 4) - 6 * np.square(x) * np.square(sigma) + 3 * np.power(sigma, 4)) / np.power(sigma, 8))) * g

		gauss0x = conv(impulse, g, axis=1)
		gauss0y = conv(impulse, g, axis=0)
		gauss0z = conv(impulse, g, axis=2)
		gauss1x = conv(impulse, g1, axis=1)
		gauss1y = conv(impulse, g1, axis=0)
		gauss1z = conv(impulse, g1, axis=2)

		for t in range(orients1):
			for f in range(orients2):
				alpha = np.cos(t * angle1) * np.sin(f * angle2)
				beta = np.sin(t * angle1) * np.sin(f * angle2)
				gamma = np.cos(f * angle2)
				# g
				hermiteBasis[i, t * orients2 + f, 0, :, :, :] = conv(conv(gauss0x, g, axis=0), g, axis=2)

				# g_x g_y g_z
				hermiteBasis[i, t * orients2 + f, 1, :, :, :] = \
					+ alpha * conv(conv(gauss0y, g, axis=2), g1, axis=1) \
					+ beta * conv(conv(gauss0x, g, axis=2), g1, axis=0) \
					+ gamma * conv(conv(gauss0x, g, axis=0), g1, axis=2)

				# g_xx g_yy g_zz g_xy g_xy g_yz
				hermiteBasis[i, t * orients2 + f, 2, :, :] = \
					+ np.power(alpha, 2) * conv(conv(gauss0y, g, axis=2), g2, axis=1) \
					+ np.power(beta, 2) * conv(conv(gauss0x, g, axis=2), g2, axis=0) \
					+ np.power(gamma, 2) * conv(conv(gauss0x, g, axis=0), g2, axis=2) \
					+ 2 * alpha * beta * conv(conv(gauss0z, g1, axis=1), g1, axis=0) \
					+ 2 * alpha * gamma * conv(conv(gauss0y, g1, axis=1), g1, axis=2) \
					+ 2 * beta * gamma * conv(conv(gauss0x, g1, axis=0), g1, axis=2)

				# g_xxx g_yyy g_zzz g_xxy g_xxz g_yyx g_yyz g_xzz g_yzz g_xyz
				hermiteBasis[i, t * orients2 + f, 3, :, :] = \
					+ np.power(alpha, 3) * conv(conv(gauss0y, g, axis=2), g3, axis=1) \
					+ np.power(beta, 3) * conv(conv(gauss0x, g, axis=2), g3, axis=0) \
					+ np.power(gamma, 3) * conv(conv(gauss0x, g, axis=0), g3, axis=2) \
					+ 3 * np.power(alpha, 2) * beta * conv(conv(gauss1y, g, axis=2), g2, axis=1) \
					+ 3 * np.power(alpha, 2) * gamma * conv(conv(gauss1z, g, axis=0), g2, axis=1) \
					+ 3 * np.power(beta, 2) * alpha * conv(conv(gauss1x, g, axis=2), g2, axis=0) \
					+ 3 * np.power(beta, 2) * gamma * conv(conv(gauss1z, g, axis=1), g2, axis=0) \
					+ 3 * np.power(gamma, 2) * alpha * conv(conv(gauss1x, g, axis=0), g2, axis=2) \
					+ 3 * np.power(gamma, 2) * beta * conv(conv(gauss1y, g, axis=1), g2, axis=2) \
					+ 6 * alpha * beta * gamma * conv(conv(gauss1y, g1, axis=1), g1, axis=2)

		# g_xxxx g_yyyy g_zzzz g_xxxy g_xxxz g_xyyy g_yyyz g_xzzz g_yzzz g_xxyy g_xxzz g_yyzz g_xxyz g_xyyz g_xyzz
		#            hermiteBasis[i, t, 4, :, :] = \
		#               np.power(np.cos(t * angle), 4) * conv(gauss0y, g4, axis=1) \
		#                + 4 * np.power(np.cos(t * angle), 3) * np.cos(np.pi / 2.0 - t * angle) * conv(conv(impulse, g3, axis=1),
		#                                                                                              g1, axis=0) \
		#                + 12 * np.power(np.cos(t * angle), 2) * np.power(np.cos(np.pi / 2.0 - t * angle), 2) * conv(
		#                    conv(impulse, g2, axis=1), g2, axis=0) \
		#                + 4 * np.cos(t * angle) * np.power(np.cos(np.pi / 2.0 - t * angle), 3) * conv(conv(impulse, g3, axis=0),
		#                                                                                              g1, axis=1) \
		#                + np.power(np.cos(np.pi / 2.0 - t * angle), 4) * conv(gauss0x, g4, axis=0)

		hermiteBasis[i, :, :, :, :, :] /= np.sqrt(sigma)

	with tf.device('/cpu:0'):
		return tf.constant(hermiteBasis[:, :, 0:order + 1, :, :, :], dtype=tf.float32)



def init_alphas(nrFilters, channels, nrBasis, name):
	with tf.device('/cpu:0'):
		return tf.get_variable(
			name,
			shape=[nrFilters, channels, nrBasis],
			initializer=tf.random_uniform_initializer(-1.0, 1.0),
			dtype=tf.float32)

def init_biases(channels, name):
	with tf.device('/cpu:0'):
		return tf.get_variable(
			name, shape=[channels],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)

def rms(x):
	return np.sqrt(np.mean(np.square(x)))


def batch_norm(_input, is_training, momentum, renorm):
	output = tf.contrib.layers.batch_norm(
		_input, decay=momentum, is_training=is_training, center=False,
		renorm=True,
		renorm_decay=renorm)  # , param_regularizers={'beta': tf.contrib.layers.l2_regularizer(self.beta_wd)})
	return output