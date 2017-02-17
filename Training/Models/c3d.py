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

class C3D(object):
	def __init__(self,
							num_classes=1000,
							is_training=True,
							dropout_keep_prob=0.5,
							spatial_squeeze=True,
							scope='c3d'):
							
		self.num_classes			= num_classes
		self.dropout_keep_prob		= dropout_keep_prob
		self.is_training			= is_training
		self.scope					= scope
		self.spatial_squeeze		= spatial_squeeze

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

	def inference(self, inputs):							
		with variable_scope.variable_scope(scope, 'c3d', [inputs]) as sc:
			end_points_collection = sc.original_name_scope + '_end_points'
			
			print(inputs.get_shape())
			# Collect outputs for conv3d, fully_connected and max_pool3d.
			
			# --- CONV1 ---
			c = inputs.get_shape()[4]
			net = _conv_layer(inputs, [3, 3, 3, c, 64], [1, 1, 1, 1, 1],
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005),
				padding='SAME', name='conv1'
			)
			net = tf.nn.max_pool3d(net, [1, 2, 2, 1, 1], [1, 2, 2, 1, 1], 
				padding='VALID', name='pool1')
			print(net.get_shape())
			
			# --- CONV2 ---
			net = _conv_layer(net, [3, 3, 3, 64, 128], [1, 1, 1, 1, 1],
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005),
				padding='SAME', name='conv2'
			)
			net = tf.nn.max_pool3d(net, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 
				padding='VALID', name='pool2')
			print(net.get_shape())
			
			# --- CONV3 ---
			net = _conv_layer(net, [3, 3, 3, 128, 256], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv3a'
			)
			net = _conv_layer(net, [3, 3, 3, 256, 256], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv3b'
			)	
			net = tf.nn.max_pool3d(net, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 
				padding='VALID', name='pool3')
			print(net.get_shape())
			
			# --- CONV4 ---
			net = _conv_layer(net, [3, 3, 3, 256, 512], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv4a'
			)
			net = _conv_layer(net, [3, 3, 3, 512, 512], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv4b'
			)	
			net = tf.nn.max_pool3d(net, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 
				padding='VALID', name='pool4')
			print(net.get_shape())
			
			# --- CONV5 ---
			net = _conv_layer(net, [3, 3, 3, 512, 512], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv5a'
			)
			net = _conv_layer(net, [3, 3, 3, 512, 512], [1, 1, 1, 1, 1], 
				w_init = tf.contrib.layers.xavier_initializer(), 
				b_init = init_ops.constant_initializer(0.1), 
				w_reg = regularizers.l2_regularizer(0.0005), 
				padding='SAME', name='conv5b'
			)	
			net = tf.nn.max_pool3d(net, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 
				padding='VALID', name='pool5')
			
			print(net.get_shape())
			# Use conv3d instead of fully_connected layers.
			s = net.get_shape()
			w = s[1].value
			h = s[2].value
			d = s[3].value
			net = _conv_layer(net, [w, h, d, 512, 4096], [1, 1, 1, 1, 1], 
				w_init=trunc_normal(0.005), 
				b_init = init_ops.constant_initializer(0.1),
				w_reg = None,			
				padding='VALID', name='fc6'
			)
			print(net.get_shape())		
			net = layers_lib.dropout(
				net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout6')
				
			net = _conv_layer(net, [1, 1, 1, 4096, 4096], [1, 1, 1, 1, 1], 
				w_init=trunc_normal(0.005), 
				b_init = init_ops.constant_initializer(0.1),
				w_reg = None, 
				padding='SAME', name='fc7'
			)
			print(net.get_shape())		
			net = layers_lib.dropout(
				net, self.dropout_keep_prob, is_training=self.is_training, scope='dropout7')
			
			net = _last_conv_layer(net, [1, 1, 1, 4096, self.num_classes], [1, 1, 1, 1, 1], 
				w_init=trunc_normal(0.005), 
				b_init = init_ops.constant_initializer(0.0), 
				w_reg = None,
				padding='SAME', name='fc8'
			)
			print(net.get_shape())
			
			# Convert end_points_collection into a end_point dict.
			end_points = utils.convert_collection_to_dict(end_points_collection)
			if self.spatial_squeeze:
				net = array_ops.squeeze(net, name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
			
			return net, end_points
