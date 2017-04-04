# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
											activation_fn=tf.nn.relu,
											biases_initializer=tf.constant_initializer(0.1),
											weights_regularizer=slim.l2_regularizer(weight_decay)):
		with slim.arg_scope([slim.conv2d], padding='SAME'):
			with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
				return arg_sc


class Alexnet(object):
	
	def __init__(self, 
						kernels_3d=False,
						num_classes=1000,
						is_training=True,
						dropout_keep_prob=0.5,
						spatial_squeeze=True,
						scope='Alexnet_v2'):
						
		self.num_classes			= num_classes
		self.dropout_keep_prob		= dropout_keep_prob
		self.is_training			= is_training

		self.inference				= self.inference_3d if kernels_3d else self.inference_2d
		self.scope					= scope
		self.spatial_squeeze		= spatial_squeeze
		
	def _conv_layer_3d(self, input, shape, stride, padding, w_init, b_init, w_reg, name):
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
		
	def _last_conv_layer_3d(self, input, shape, stride, padding, w_init, b_init, w_reg, name):
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
	
	def inference_2d(self, inputs):
		with tf.variable_scope(self.scope, 'Alexnet_v2', [inputs]) as sc:
			end_points_collection = sc.name + '_end_points'
			# Collect outputs for conv2d, fully_connected and max_pool2d.
			with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
													outputs_collections=[end_points_collection]):
				print(inputs.get_shape())
				net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='Conv1')
				print(net.get_shape())
				net = slim.max_pool2d(net, [3, 3], 2, scope='Pool1')
				print(net.get_shape())
				net = slim.conv2d(net, 192, [5, 5], scope='Conv2')
				print(net.get_shape())
				net = slim.max_pool2d(net, [3, 3], 2, scope='Pool2')
				print(net.get_shape())
				net = slim.conv2d(net, 384, [3, 3], scope='Conv3')
				print(net.get_shape())
				net = slim.conv2d(net, 384, [3, 3], scope='Conv4')
				print(net.get_shape())
				net = slim.conv2d(net, 256, [3, 3], scope='Conv5')
				print(net.get_shape())
				net = slim.max_pool2d(net, [3, 3], 2, scope='Pool5')
				print(net.get_shape())
				
				# Use conv2d instead of fully_connected layers.
				k = net.get_shape()[1].value
				with slim.arg_scope([slim.conv2d],
														weights_initializer=trunc_normal(0.005),
														biases_initializer=tf.constant_initializer(0.1)):
					net = slim.conv2d(net, 4096, [k, k], padding='VALID',
														scope='FC6')
					print(net.get_shape())
					net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,
														 scope='Dropout6')
					print(net.get_shape())
					net = slim.conv2d(net, 4096, [1, 1], scope='FC7')
					print(net.get_shape())
					net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,
														 scope='Dropout7')
					print(net.get_shape())
					net = slim.conv2d(net, self.num_classes, [1, 1],
														activation_fn=None,
														normalizer_fn=None,
														biases_initializer=tf.zeros_initializer,
														scope='FC8')
					print(net.get_shape())

				# Convert end_points_collection into a end_point dict.
				end_points = slim.utils.convert_collection_to_dict(end_points_collection)
				if self.spatial_squeeze:
					net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
					end_points[sc.name + '/fc8'] = net
				
				return net
				
	def inference_3d(self, inputs):
		
		end_points_collection = sc.original_name_scope + '_end_points'
		
		print(inputs.get_shape())
		# Collect outputs for conv3d, fully_connected and max_pool3d.
		c = inputs.get_shape()[4]
		net = self._conv_layer_3d(inputs, [11, 11, 7, c, 64], [1, 4, 4, 1, 1],
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005),
			padding='VALID', name='conv1'
		)
		print(net.get_shape())
		net = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], [1, 2, 2, 1, 1], 
			padding='VALID', name='pool1')
		print(net.get_shape())
		net = self._conv_layer_3d(net, [5, 5, 5, 64, 192], [1, 1, 1, 1, 1],
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005),
			padding='SAME', name='conv2'
		)
		print(net.get_shape())
		net = tf.nn.max_pool3d(net, [1, 3, 3, 2, 1], [1, 2, 2, 2, 1], 
			padding='VALID', name='pool2')
		print(net.get_shape())
		net = self._conv_layer_3d(net, [3, 3, 3, 192, 384], [1, 1, 1, 1, 1], 
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005), 
			padding='SAME', name='conv3'
		)
		print(net.get_shape())
		net = self._conv_layer_3d(net, [3, 3, 3, 384, 384], [1, 1, 1, 1, 1], 
			w_init = tf.contrib.layers.xavier_initializer(), 
			b_init = init_ops.constant_initializer(0.1), 
			w_reg = regularizers.l2_regularizer(0.0005), 
			padding='SAME', name='conv4'
		)	
		print(net.get_shape())
		net = self._conv_layer_3d(net, [3, 3, 3, 384, 256], [1, 1, 1, 1, 1], 
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
		s = net.get_shape()
		w = s[1].value
		h = s[2].value
		d = s[3].value
		net = self._conv_layer_3d(net, [w, h, d, 256, 4096], [1, 1, 1, 1, 1], 
			w_init=trunc_normal(0.005), 
			b_init = init_ops.constant_initializer(0.1),
			w_reg = None,			
			padding='VALID', name='fc6'
		)
		print(net.get_shape())		
		net = layers_lib.dropout(
			net, dropout_keep_prob, is_training=is_training, scope='dropout6')
			
		net = self._conv_layer_3d(net, [1, 1, 1, 4096, 4096], [1, 1, 1, 1, 1], 
			w_init=trunc_normal(0.005), 
			b_init = init_ops.constant_initializer(0.1),
			w_reg = None, 
			padding='SAME', name='fc7'
		)
		print(net.get_shape())		
		net = layers_lib.dropout(
			net, dropout_keep_prob, is_training=is_training, scope='dropout7')
		
		net = self._last_conv_layer_3d(net, [1, 1, 1, 4096, num_classes], [1, 1, 1, 1, 1], 
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
		
		return net
	