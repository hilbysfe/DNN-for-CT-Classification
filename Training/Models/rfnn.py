from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from Utils.rfnn_utils import init_alphas
from Utils.rfnn_utils import init_biases
from Utils.rfnn_utils import init_basis_hermite_2D
from Utils.rfnn_utils import init_basis_hermite_3D
from Utils.rfnn_utils import _rfnn_conv_layer_2d

from Utils.cnn_utils import batch_norm_wrapper 
from Utils.cnn_utils import _conv_layer_2d_with_kernel_and_bias


	
class RFNN(object):

	def __init__(self, sigmas, kernels, maps, n_classes=2, bases_3d = False, dropout_rate_conv=.2, dropout_rate_hidden=0.7, is_training = True):
	
		self.n_classes				= n_classes
		self.dropout_rate_conv		= dropout_rate_conv
		self.dropout_rate_hidden	= dropout_rate_hidden
		self.is_training			= is_training
		
		self.act					= tf.nn.relu
		self.regularizer 			= tf.contrib.layers.l2_regularizer(0.0005)
				
		self.hermit					= init_basis_hermite_3D if bases_3d else init_basis_hermite_2D		
		self.inference				= self.inference_3d if bases_3d else self.inference_2d
		
		self.sigmas					= sigmas
		self.maps					= maps
		
		#-------------------------
		# Init Basis and Alphas
		#-------------------------

		bases_L1 = 10
		bases_L2 = 6
		bases_L3 = 6
				
		self.basis_L1 = self.hermit(kernels[0], sigmas, bases_L1)
		self.basis_L2 = self.hermit(kernels[1], sigmas, bases_L2)
		self.basis_L3 = self.hermit(kernels[2], sigmas, bases_L3)	
	


	#---------------------------
	# Model Definition
	#---------------------------

	# ---- Use this for 2D models ----
	def inference_2d(self, X):
		
		# --- Define forward pass ---
		print(X.get_shape())
		
		# ==== Layer 1 ====
		with tf.variable_scope('ConvLayer1'):
			self.alphas_L1, self.biases_L1, layer1, w_L1 = _rfnn_conv_layer_2d(
								input=X,
								basis=self.basis_L1,
								omaps=self.maps[0],
								strides=[1,2,2,1],
								padding='SAME',
								is_training=self.is_training,
								bnorm=False)
		
		# ==== MaxPool1 ====
		# net = [ tf.nn.max_pool(layer1[i], ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID")
			# for i in range(np.shape(self.sigmas)[0]) ]
		# print(net.get_shape())
		
		# --- maxpool along scale and rotation
		# net = tf.stack(layer1)
		# layer1 = tf.stack( [ tf.reduce_max(layer1[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L1.get_shape()[0].value) ]
						# )

		# net = tf.reshape(layer1, [-1, layer1.get_shape()[2].value,
								  # layer1.get_shape()[3].value, layer1.get_shape()[0].value])
		net = tf.stack(layer1)
		net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
		print(net.get_shape())
		
		# ==== Layer 2a ====	
		with tf.variable_scope('ConvLayer2a'):		
			self.alphas_L2, self.biases_L2, layer2, w_L2 = _rfnn_conv_layer_2d(
						input=net,
						basis=self.basis_L2,
						omaps=self.maps[1],
						strides=[1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=False)
		
		# --- maxpool along scale and rotation		
		# layer2 = tf.stack( [ tf.reduce_max(layer2[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L2.get_shape()[0].value) ]
						# )
		# net = tf.reshape(layer2, [-1, layer2.get_shape()[2].value,
								  # layer2.get_shape()[3].value, layer2.get_shape()[0].value])
		layer2 = tf.stack(layer2)
		net = tf.stack(tf.reduce_max(layer2, reduction_indices=[0]))
		print(net.get_shape())
		
		# ==== Layer 2b ====		
		with tf.variable_scope('ConvLayer2b'):		
			self.alphas_L3, self.biases_L3, layer3, w_L3 = _rfnn_conv_layer_2d(
						input=net,
						basis=self.basis_L3,
						omaps=self.n_classes,
						strides=[1,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=False)
			
		# ==== MaxPool2 ====
		# net = [ tf.nn.max_pool(layer3[i], ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID") 
			# for i in range(np.shape(self.sigmas)[0]) ]
		# print(net.get_shape())
		
		# --- maxpool along scale and rotation
		# layer3 = tf.stack( [ tf.reduce_max(layer3[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L3.get_shape()[0].value) ]
						# )
		# net = tf.reshape(layer3, [-1, layer3.get_shape()[2].value,
								  # layer3.get_shape()[3].value, layer3.get_shape()[0].value])
		net = tf.stack(layer3)
		net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
		print(net.get_shape())
		
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
		
		
		
		
		
	# ---- Use this for 3D models ----
	def inference_3d(self, X):
		# --- Define forward pass ---
		print(X.get_shape())
		
		# ==== Layer 1 ====
		with tf.variable_scope('ConvLayer1'):
			self.alphas_L1, self.biases_L1, layer1, w_L1 = _rfnn_conv_layer_3d(
								input=X,
								basis=self.basis_L1,
								omaps=self.maps[0],
								strides=[1,2,2,2,1],
								padding='SAME',
								is_training=self.is_training,
								bnorm=False)
		
		# ==== MaxPool1 ====
		# net = [ tf.nn.max_pool(layer1[i], ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID")
			# for i in range(np.shape(self.sigmas)[0]) ]
		# print(net.get_shape())
		
		# --- maxpool along scale and rotation
		# net = tf.stack(layer1)
		# layer1 = tf.stack( [ tf.reduce_max(layer1[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L1.get_shape()[0].value) ]
						# )

		# net = tf.reshape(layer1, [-1, layer1.get_shape()[2].value,
								  # layer1.get_shape()[3].value, layer1.get_shape()[0].value])
		net = tf.stack(layer1)
		net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
		print(net.get_shape())
		
		# ==== Layer 2a ====	
		with tf.variable_scope('ConvLayer2a'):		
			self.alphas_L2, self.biases_L2, layer2, w_L2 = _rfnn_conv_layer_3d(
						input=net,
						basis=self.basis_L2,
						omaps=self.maps[1],
						strides=[1,1,1,1,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=False)
		
		# --- maxpool along scale and rotation		
		# layer2 = tf.stack( [ tf.reduce_max(layer2[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L2.get_shape()[0].value) ]
						# )
		# net = tf.reshape(layer2, [-1, layer2.get_shape()[2].value,
								  # layer2.get_shape()[3].value, layer2.get_shape()[0].value])
		net = tf.stack(layer2)
		net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
		print(net.get_shape())
		
		# ==== Layer 2b ====		
		with tf.variable_scope('ConvLayer2b'):		
			self.alphas_L3, self.biases_L3, layer3, w_L3 = _rfnn_conv_layer_3d(
						input=net,
						basis=self.basis_L3,
						omaps=self.n_classes,
						strides=[1,2,2,2,1],
						padding='SAME',
						is_training=self.is_training,
						bnorm=False)
			
		# ==== MaxPool2 ====
		# net = [ tf.nn.max_pool(layer3[i], ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID") 
			# for i in range(np.shape(self.sigmas)[0]) ]
		# print(net.get_shape())
		
		# --- maxpool along scale and rotation
		# layer3 = tf.stack( [ tf.reduce_max(layer3[:,:,:,:,i], reduction_indices=[0])
							# for i in range(self.alphas_L3.get_shape()[0].value) ]
						# )
		# net = tf.reshape(layer3, [-1, layer3.get_shape()[2].value,
								  # layer3.get_shape()[3].value, layer3.get_shape()[0].value])
		net = tf.stack(layer3)
		net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
		print(net.get_shape())
		
		# ==== AVG Pooling ====		
		k = net.get_shape()[1].value
		j = net.get_shape()[3].value
		net = tf.nn.avg_pool3d(net, ksize=[1,k,k,j,1], strides=[1,1,1,1,1], padding="VALID")
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
