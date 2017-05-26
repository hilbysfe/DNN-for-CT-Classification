from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from Models.ae import Autoencoder
import cifar10_utils

from Utils.cnn_utils import batch_norm_wrapper
from Utils.cnn_utils import _conv_layer_2d


CHECKPOINT_PATH = './ae_models/model.ckpt.meta'


class CTNET(object):

	def __init__(self, n_classes, kernels, maps, maxpool_kernels, pretraining, l2=.0, dropout_rate_conv=.0, dropout_rate_hidden=.0, is_training = True, conv3d=False):
	
		self.n_classes				= n_classes
		self.is_training			= is_training
		self.pretraining			= pretraining
		
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
		
		
	# ---- Use this for 2D models ----
	def inference_2d(self, X):
		print(X.get_shape())
		ch = X.get_shape()[3].value
		net = X
		# ==== Layer 1 ====			
		if not self.pretraining:
			with tf.variable_scope('ConvLayer1'):
				net,_ = _conv_layer_2d(
						input=net,
						shape=[self.kernels[0], self.kernels[0], ch, self.maps[0]],
						strides=[1,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=True)
				print(net.get_shape())

		# if self.dropout_rate_conv > 0.0:
			# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			with tf.variable_scope('ConvLayer2'):
				net,_ = _conv_layer_2d(
						input=net,
						shape=[self.kernels[1], self.kernels[1], self.maps[0], self.maps[1]],
						strides=[1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=True)
				print(net.get_shape())
			
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			with tf.variable_scope('ConvLayer3'):
				net,_ = _conv_layer_2d(
						input=net,
						shape=[self.kernels[2], self.kernels[2], self.maps[1], self.n_classes],
						strides=[1,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=True)
				print(net.get_shape())
					
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
							
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
		
		return pyx
		
	def inference_3d(self, X):
		print(X.get_shape())
		# ch = X.get_shape()[3].value
				
		# ==== Layer 1 ====				
		with tf.variable_scope('ConvLayer1'):
			net = self._conv_layer_3d(
					input=X,
					shape=[self.kernels[0], self.kernels[0], self.kernels[1], 1, self.maps[0]],
					strides=[1,1,1,1,1],
					padding='SAME',
					is_training=self.is_training,
					bnorm=True)
		with tf.variable_scope('MaxPool1'):
			net = tf.nn.max_pool3d(net, ksize=[1,self.mp_kernels[0],self.mp_kernels[0],self.mp_kernels[0],1], strides=[1,2,2,2,1], padding="VALID")
			print(net.get_shape())

		if self.dropout_rate_conv > 0.0:
			keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			with tf.variable_scope('ConvLayer2'):
				net = self._conv_layer_3d(
						input=net,
						shape=[self.kernels[2], self.kernels[2], self.kernels[3], self.maps[0], self.maps[1]],
						strides=[1,1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=True)

		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			with tf.variable_scope('ConvLayer3'):
				net = self._conv_layer_3d(
						input=net,
						shape=[self.kernels[4], self.kernels[4], self.kernels[5], self.maps[1], self.maps[2]],
						strides=[1,1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=True)
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
