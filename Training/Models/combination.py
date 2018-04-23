
import numpy as np
import tensorflow as tf


class CombinationModel(object):
	def __init__(self, embedding_neurons, num_of_layers, num_of_neurons, keep_prob, is_training):

		self.num_of_layers = num_of_layers
		self.num_of_neurons = num_of_neurons
		self.embedding_neurons = embedding_neurons
		self.is_training = is_training
		self.keep_prob = keep_prob

	def inference(self, model_outputs):
		with tf.variable_scope("Combination_model"):
			with tf.variable_scope("Embedding_layer"):
				outputs = []
				for i, output in enumerate(model_outputs):
					weights = tf.get_variable(
						'weights_'+str(i),
						shape=[output.get_shape()[-1].value, self.embedding_neurons],
#						initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / self.embedding_neurons),
#																	dtype=tf.float32),
						initializer=tf.contrib.layers.xavier_initializer(),
						dtype=tf.float32)
					biases = tf.get_variable(
						'biases_'+str(i),
						shape=[self.embedding_neurons],
						initializer=tf.constant_initializer(0.0),
						dtype=tf.float32)
					# Embedding to X dimensions
					embedded = tf.nn.bias_add(tf.matmul(output, weights), biases)
					# Normalization without Scale and Center
					normalized = tf.contrib.layers.batch_norm(embedded, decay=0.9,
									is_training=self.is_training, center=False, scale=False)
					outputs.append(normalized)
			with tf.variable_scope("Concatenation_layer"):
				# Concat into 1 feature vector
				output = tf.concat(axis=1, values=outputs)

			with tf.variable_scope("Classifier"):
				# Add fully connected layers
				for layer in range(self.num_of_layers):
					output = tf.nn.sigmoid(output, name="Activation_" + str(layer))

					# Dropout
					if layer != 0:
						output = tf.cond(
							self.is_training,
							lambda: tf.nn.dropout(output, self.keep_prob),
							lambda: output
						)

					with tf.variable_scope("Layer_" + str(layer)):
						weights = tf.get_variable(
							'weights',
							shape=[len(model_outputs)*self.embedding_neurons if layer==0 else self.num_of_neurons[layer-1],
								   self.num_of_neurons[layer]],
#							initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / self.num_of_neurons[layer]), dtype=tf.float32),
							initializer=tf.contrib.layers.xavier_initializer(),
							dtype=tf.float32)

						biases = tf.get_variable(
							'biases',
							[self.num_of_neurons[layer]],
							initializer=tf.constant_initializer(0.0),
							dtype=tf.float32)

						output = tf.nn.bias_add(tf.matmul(output, weights), biases)

				with tf.variable_scope("Output"):
					output = tf.nn.sigmoid(output, name="Activation_pre-output")
					# Dropout
					output = tf.cond(
						self.is_training,
						lambda: tf.nn.dropout(output, self.keep_prob),
						lambda: output
					)
					weights = tf.get_variable(
						'weights',
						shape=[output.get_shape()[-1],
							   2],
#						initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2 / 2), dtype=tf.float32),
						initializer=tf.contrib.layers.xavier_initializer(),
						dtype=tf.float32)

					biases = tf.get_variable(
						'biases',
						[2],
						initializer=tf.constant_initializer(0.0),
						dtype=tf.float32)

					output = tf.nn.bias_add(tf.matmul(output, weights), biases)

		return output



