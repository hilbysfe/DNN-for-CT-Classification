from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from Models.ae import Autoencoder

from Utils.cnn_utils import batch_norm_wrapper
from Utils.cnn_utils import _conv_layer_2d
from Utils.cnn_utils import _conv_layer_3d
from Utils.cnn_utils import _softmax_layer


CHECKPOINT_PATH = './ae_models/model.ckpt.meta'


class CTNET(object):

	def __init__(self, n_classes, kernels, maps, strides, pretraining=False, l2=.0, dropout_rate_conv=.0, dropout_rate_hidden=.0, is_training = True, conv3d=False, bnorm=False):
	
		self.n_classes				= n_classes
		self.is_training			= is_training
		self.pretraining			= pretraining
		self.bnorm					= bnorm
		
		self.dropout_rate_conv		= dropout_rate_conv
		self.dropout_rate_hidden	= dropout_rate_hidden
		
		self.act					= tf.nn.relu
		self.kernels				= kernels
		self.maps					= maps
		self.strides				= strides
		
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
				net = _conv_layer_2d(
						input=net,
						shape=[self.kernels[0], self.kernels[0], ch, self.maps[0]],
						strides=[1,self.strides[0],self.strides[0],1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())

		# if self.dropout_rate_conv > 0.0:
			# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			with tf.variable_scope('ConvLayer2'):
				net = _conv_layer_2d(
						input=net,
						shape=[self.kernels[1], self.kernels[1], self.maps[0], self.maps[1]],
						strides=[1,self.strides[1],self.strides[1],1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())
			
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			with tf.variable_scope('ConvLayer3'):
				net = _conv_layer_2d(
						input=net,
						shape=[self.kernels[2], self.kernels[2], self.maps[1], self.maps[2]],
						strides=[1,self.strides[2],self.strides[2],1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())
					
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
		# ==== FULL Layer ====
#		fshape = net.get_shape()
#		dim = fshape[1].value*fshape[2].value*fshape[3].value
#		net = tf.reshape(net, [-1, dim], name='Activation')
		
#		print(net.get_shape())
		
#		pyx = _softmax_layer(
#				input=net,
#				shape=[dim, self.n_classes],
#				is_training=self.is_training,
#				bnorm=False
#		)
		
		# ==== AVG Pooling ====		
		k = net.get_shape()[1].value
		net = tf.nn.avg_pool(net, ksize=[1,k,k,1], strides=[1,1,1,1], padding="VALID")
		
		
		# ==== Flatten ====		
		with tf.variable_scope('Flatten'):
			fshape = net.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value
			pyx = tf.reshape(net, [-1, dim])
		
		print(pyx.get_shape())			
		
		return pyx
		
	def inference_3d(self, X):
		print(X.get_shape())
		net = X
		# ==== Layer 1 ====			
		if not self.pretraining:
			with tf.variable_scope('ConvLayer1'):
				net = _conv_layer_3d(
						input=net,
						shape=[self.kernels[0], self.kernels[0], self.kernels[0], 1, self.maps[0]],
						strides=[1,2,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())

		# if self.dropout_rate_conv > 0.0:
			# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
			# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 2 ====			
		if len(self.kernels) > 1:
			with tf.variable_scope('ConvLayer2'):
				net = _conv_layer_3d(
						input=net,
						shape=[self.kernels[1], self.kernels[1], self.kernels[1], self.maps[0], self.maps[1]],
						strides=[1,1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())
			
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
		# ==== Layer 3 ====			
		if len(self.kernels) > 2:
			with tf.variable_scope('ConvLayer3'):
				net = _conv_layer_3d(
						input=net,
						shape=[self.kernels[2], self.kernels[2], self.kernels[2], self.maps[1], self.n_classes],
						strides=[1,2,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=self.bnorm)
				print(net.get_shape())
					
			# if self.dropout_rate_conv > 0.0:
				# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
				# net = tf.nn.dropout(net, keep_prob)
		
							
		# ==== AVG Pooling ====		
		k = net.get_shape()[1].value
		j = net.get_shape()[3].value
		net = tf.cast(tf.nn.avg_pool3d(tf.cast(net, tf.float32), ksize=[1,k,k,j,1], strides=[1,1,1,1,1], padding="VALID"), tf.float32)
		print(net.get_shape())
		
		# ==== Flatten ====		
		with tf.variable_scope('Flatten'):
			fshape = net.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value*fshape[4].value
			pyx = tf.reshape(net, [-1, dim])
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
