from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class VGGConvNet(object):
	"""
	This class implements a convolutional neural network in TensorFlow.
	It incorporates a certain graph model to be trained and to be used
	in inference.
	"""

	def __init__(self, n_classes = 10, is_training = True, is_refine = False):
		"""
		Constructor for an ConvNet object. Default values should be used as hints for
		the usage of each parameter.
		Args:
		  n_classes: int, number of classes of the classification problem.
						  This number is required in order to specify the
						  output dimensions of the ConvNet.
		"""
		
		self.n_classes			= n_classes
		self.act				= tf.nn.relu
		self.regularizer =		tf.contrib.layers.l2_regularizer(0.005)
#		self.regularizer =		None
#		self.initializer =		tf.contrib.layers.xavier_initializer()
		self.initializer =		tf.truncated_normal_initializer(stddev=1e-3,dtype=tf.float32)

		self.dropout_rate		= 0.4
#		self.normalizer			= tf.nn.lrn
		self.normalizer			= None
		self.is_training		= is_training
		self.is_refine			= is_refine
		
	def _activation_summary(self, x):
		tensor_name = x.op.name
		tf.histogram_summary(tensor_name, x)
		
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
		
	def _conv_layer(self, input, shape, stride, padding, name):
		with tf.variable_scope(name) as scope:
			kernel = tf.get_variable(
				'weights',
				shape,
				initializer=self.initializer)
			biases = tf.get_variable(
				'biases',
				[shape[-1]],
				initializer=tf.constant_initializer(0.0))
			
			conv = tf.nn.conv2d(input, kernel, strides=stride, padding=padding)
			conv_out = self.act(tf.nn.bias_add(conv, biases), name=scope.name)
			
			self._activation_summary(conv_out)
			
		return conv_out
			
	def _pool_layer(self, input, kernel, stride, padding, name):
		with tf.variable_scope(name) as scope:
			pool = tf.nn.max_pool(
				input,
				ksize=kernel,
				strides=stride,
                padding=padding,
				name="Activation")
			
			self._activation_summary(pool)
			
		return pool
		
	def _full_layer(self, input, shape, name):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights', 
				shape=shape,
				initializer=self.initializer,
				regularizer=self.regularizer)
			biases = tf.get_variable(
				'biases', 
				shape=[shape[-1]], 
				initializer=tf.constant_initializer(0.0))
				
			local = self.act(tf.add(tf.matmul(input, weights), biases), name="Activation")
			
			self._activation_summary(local)
			
		return local
		
	def _softmax_layer(self, input, shape, name):
		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights', 
				shape=shape,
				initializer=self.initializer,
				regularizer=self.regularizer)
			biases = tf.get_variable(
				'biases', 
				shape=[shape[-1]], 
				initializer=tf.constant_initializer(0.0))
				
			local = tf.add(tf.matmul(input, weights), biases, name="Activation")
			
			self._activation_summary(local)
			
		return local
		
	def _dropout_layer(self, input, drop_rate, name):
		with tf.variable_scope(name):
			keep_prob = tf.select(self.is_training, 1-drop_rate, 1)
			valid_keep = tf.select(self.is_refine, keep_prob, 1)
			logits = tf.nn.dropout(input, valid_keep, name='Activation')
			
		return logits
		
	def inference(self, x):
		"""
		Performs inference given an input tensor. This is the central portion
		of the network where we describe the computation graph. Here an input
		tensor undergoes a series of convolution, pooling and nonlinear operations
		as defined in this method. For the details of the model, please
		see assignment file.

		Here we recommend you to consider using variable and name scopes in order
		to make your graph more intelligible for later references in TensorBoard
		and so on. You can define a name scope for the whole model or for each
		operator group (e.g. conv+pool+relu) individually to group them by name.
		Variable scopes are essential components in TensorFlow for parameter sharing.
		Although the model(s) which are within the scope of this class do not require
		parameter sharing it is a good practice to use variable scope to encapsulate
		model.

		Args:
		  x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

		Returns:
		  logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
				  the logits outputs (before softmax transformation) of the
				  network. These logits can then be used with loss and accuracy
				  to evaluate the model.
		"""
		with tf.variable_scope('VGGConvNet'):
			########################
			# PUT YOUR CODE HERE  #
			########################
			
			# flatten
			with tf.variable_scope('Flatten'):
#				fshape = x.get_shape()
#				dim = fshape[1].value*fshape[2].value*fshape[3].value
				flatten = tf.reshape(x, [-1, 512], name='Activation')			
			
			# full1
			full1 = self._full_layer(
						input=flatten,
						shape=[512, 384], 
						name='FullLayer1'
						)
			# dropout1
#			drop1 = self._dropout_layer(
#						input=full1,
#						drop_rate = self.dropout_rate,
#						name='DropoutLayer1')
						
			# full2
			full2 = self._full_layer(
						input=full1,
						shape=[384, 192],
						name='FullLayer2'
						)
			
			# dropout2
#			drop2 = self._dropout_layer(
#						input=full2,
#						drop_rate = self.dropout_rate,
#						name='DropoutLayer2')
			
			# softmax linear
			logits = self._softmax_layer(
						input=full2,
						shape=[192, self.n_classes],
						name='SoftmaxLayer1'
						)
			
			########################
			# END OF YOUR CODE    #
			########################
		return logits

	def accuracy(self, logits, labels):
		"""
		Calculate the prediction accuracy, i.e. the average correct predictions
		of the network.
		As in self.loss above, you can use tf.scalar_summary to save
		scalar summaries of accuracy for later use with the TensorBoard.

		Args:
		  logits: 2D float Tensor of size [batch_size, self.n_classes].
					   The predictions returned through self.inference.
		  labels: 2D int Tensor of size [batch_size, self.n_classes]
					 with one-hot encoding. Ground truth labels for
					 each observation in batch.

		Returns:
		  accuracy: scalar float Tensor, the accuracy of predictions,
					i.e. the average correct predictions over the whole batch.
		"""
		########################
		# PUT YOUR CODE HERE  #
		########################
		
		softmax = tf.nn.softmax(logits)
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)
		
		########################
		# END OF YOUR CODE    #
		########################

		return accuracy

	def loss(self, logits, labels):
		"""
		Calculates the multiclass cross-entropy loss from the logits predictions and
		the ground truth labels. The function will also add the regularization
		loss from network weights to the total loss that is return.
		In order to implement this function you should have a look at
		tf.nn.softmax_cross_entropy_with_logits.
		You can use tf.scalar_summary to save scalar summaries of
		cross-entropy loss, regularization loss, and full loss (both summed)
		for use with TensorBoard. This will be useful for compiling your report.

		Args:
		  logits: 2D float Tensor of size [batch_size, self.n_classes].
					   The predictions returned through self.inference.
		  labels: 2D int Tensor of size [batch_size, self.n_classes]
					   with one-hot encoding. Ground truth labels for each
					   observation in batch.

		Returns:
		  loss: scalar float Tensor, full loss = cross_entropy + reg_loss
		"""
		########################
		# PUT YOUR CODE HERE  #
		########################
		
		with tf.variable_scope('Losses') as scope:		
			with tf.name_scope('Cross_Entropy_Loss'):
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
				cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
				
				tf.add_to_collection('losses', cross_entropy_mean)		
				tf.scalar_summary('cross_entropy', cross_entropy_mean)
			with tf.name_scope('Regularization_Loss'):
				reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
				
				tf.add_to_collection('losses', reg_loss)
				tf.scalar_summary('reg_loss', reg_loss)
			with tf.name_scope('Total_Loss'):
				loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
				
				tf.scalar_summary('total_loss', loss)
		
		########################
		# END OF YOUR CODE    #
		########################

		return loss
