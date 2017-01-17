# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.
This work was first described in:
	ImageNet Classification with Deep Convolutional Neural Networks
	Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
and later refined in:
	One weird trick for parallelizing convolutional neural networks
	Alex Krizhevsky, 2014
Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
Usage:
	with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
		outputs, end_points = alexnet.alexnet_v2(inputs)
@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

def _conv_layer(input, shape, stride, padding, w_init, b_init, w_reg, name):
	with tf.variable_scope(name) as scope:
		kernel = tf.get_variable(
			'weights',
			shape,
			initializer = w_init,
			regularizer = w_reg)
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=b_init)
		
		conv = tf.nn.conv3d(input, kernel, strides=stride, padding=padding)
		conv_out = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
		
#		self._activation_summary(conv_out)
		
	return conv_out
	
def _last_conv_layer(input, shape, stride, padding, w_init, b_init, w_reg, name):
	with tf.variable_scope(name) as scope:
		kernel = tf.get_variable(
			'weights',
			shape,
			initializer=w_init,
			regularizer = w_reg)
		biases = tf.get_variable(
			'biases',
			[shape[-1]],
			initializer=b_init)
		
		conv = tf.nn.conv3d(input, kernel, strides=stride, padding=padding)
		conv_out = tf.nn.bias_add(conv, biases, name=scope.name)
		
#		self._activation_summary(conv_out)
		
	return conv_out

def alexnet_v2(inputs,
							 num_classes=1000,
							 is_training=True,
							 dropout_keep_prob=0.5,
							 spatial_squeeze=True,
							 scope='alexnet_v2'):
	"""AlexNet version 2.
	Described in: http://arxiv.org/pdf/1404.5997v2.pdf
	Parameters from:
	github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
	layers-imagenet-1gpu.cfg
	Note: All the fully_connected layers have been transformed to conv3d layers.
				To use in classification mode, resize input to 224x224. To use in fully
				convolutional mode, set spatial_squeeze to false.
				The LRN layers have been removed and change the initializers from
				random_normal_initializer to xavier_initializer.
	Args:
		inputs: a tensor of size [batch_size, height, width, channels].
		num_classes: number of predicted classes.
		is_training: whether or not the model is being trained.
		dropout_keep_prob: the probability that activations are kept in the dropout
			layers during training.
		spatial_squeeze: whether or not should squeeze the spatial dimensions of the
			outputs. Useful to remove unnecessary dimensions for classification.
		scope: Optional scope for the variables.
	Returns:
		the last op containing the log predictions and end_points dict.
	"""
	with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		
		print(inputs.get_shape())
		# Collect outputs for conv3d, fully_connected and max_pool3d.
		net = _conv_layer(inputs, [11, 11, 7, 1, 64], [1, 4, 4, 1, 1],
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005),
			padding='VALID', name='conv1'
		)
		print(net.get_shape())
		net = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], [1, 2, 2, 1, 1], 
			padding='VALID', name='pool1')
		print(net.get_shape())
		net = _conv_layer(net, [5, 5, 5, 64, 192], [1, 1, 1, 1, 1],
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005),
			padding='SAME', name='conv2'
		)
		print(net.get_shape())
		net = tf.nn.max_pool3d(net, [1, 3, 3, 2, 1], [1, 2, 2, 2, 1], 
			padding='VALID', name='pool2')
		print(net.get_shape())
		net = _conv_layer(net, [3, 3, 3, 192, 384], [1, 1, 1, 1, 1], 
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005), 
			padding='SAME', name='conv3'
		)
		print(net.get_shape())
		net = _conv_layer(net, [3, 3, 3, 384, 384], [1, 1, 1, 1, 1], 
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005), 
			padding='SAME', name='conv4'
		)	
		print(net.get_shape())
		net = _conv_layer(net, [3, 3, 3, 384, 256], [1, 1, 1, 1, 1], 
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005), 
			padding='SAME', name='conv5'
		)
		print(net.get_shape())
		net = tf.nn.max_pool3d(net, [1, 3, 3, 2, 1], [1, 2, 2, 1, 1], 
			padding='VALID', name='pool5')
		print(net.get_shape())
		# Use conv3d instead of fully_connected layers.
		net = _conv_layer(net, [5, 5, 3, 256, 4096], [1, 1, 1, 1, 1], 
			w_init=trunc_normal(0.005), 
			b_init = init_ops.constant_initializer(0.1),
			w_reg = None,			
			padding='VALID', name='fc6'
		)
		print(net.get_shape())		
		net = layers_lib.dropout(
			net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			
		net = _conv_layer(net, [1, 1, 1, 4096, 4096], [1, 1, 1, 1, 1], 
			w_init=trunc_normal(0.005), 
			b_init = init_ops.constant_initializer(0.1),
			w_reg = None, 
			padding='SAME', name='fc7'
		)
		print(net.get_shape())		
		net = layers_lib.dropout(
			net, dropout_keep_prob, is_training=is_training, scope='dropout7')
		
		net = _last_conv_layer(net, [1, 1, 1, 4096, num_classes], [1, 1, 1, 1, 1], 
			w_init=trunc_normal(0.005), 
			b_init = init_ops.constant_initializer(0.0), 
			w_reg = None,
			padding='SAME', name='fc8'
		)
		print(net.get_shape())
		
		# Convert end_points_collection into a end_point dict.
		end_points = utils.convert_collection_to_dict(end_points_collection)
		if spatial_squeeze:
			net = array_ops.squeeze(net, name='fc8/squeezed')
			end_points[sc.name + '/fc8'] = net
		
		return net, end_points


alexnet_v2.default_image_size = 224