
import numpy as np
import tensorflow as tf


class Combination_model(object):
	def __init__(self, embedding_neurons, num_of_layers, num_of_neurons, is_training):

		self.num_of_layers = num_of_layers
		self.num_of_neurons = num_of_neurons
		self.embedding_neurons = embedding_neurons
		self.is_training = is_training


	def inference(self, model_outputs):
		with tf.variable_scope("Embedding_layer"):
			outputs = []
			for i, output in enumerate(model_outputs):
				new_shape = output.get_shape()
				new_shape[-2] = new_shape[-1]
				new_shape[-1] = self.embedding_neurons
				weights = tf.get_variable(
					'weights_'+str(i),
					shape=new_shape,
					initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / new_shape[-1]), dtype=tf.float32),
					dtype = tf.float32
				)
				# Embedding to X dimensions
				embedded = tf.matmul(output, weights)
				# Normalization without Scale and Center
				normalized = tf.contrib.layers.batch_norm(embedded,
								is_training=self.is_training, center=False, scale=False)
				outputs.append(normalized)
		with tf.variable_scope("Concatenation_layer"):
			# Concat into 1 feature vector
			output = tf.concat(axis=3, values=outputs)

		with tf.variable_scope("Classifier"):
			# Add fully connected layers
			for layer in range(self.num_of_layers):
				output = tf.nn.tanh(output, name="Activation_" + str(layer))

				new_shape = output.get_shape()
				new_shape[-2] = new_shape[-1]
				new_shape[-1] = self.num_of_neurons[layer]
				with tf.variable_scope("Layer_" + str(layer)):
					weights = tf.get_variable(
						'weights',
						shape=new_shape,
						initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / new_shape[-1]), dtype=tf.float32),
						dtype=tf.float32)

					biases = tf.get_variable(
						'biases',
						[new_shape[-1]],
						initializer=tf.constant_initializer(0.0),
						dtype=tf.float32)

					output = tf.nn.bias_add(tf.matmul(output, weights), biases)

		return output



