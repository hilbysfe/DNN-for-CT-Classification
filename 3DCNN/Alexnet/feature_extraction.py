from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from convnet import ConvNet
from siamese import Siamese


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 4
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10

LOG_DIR_DEFAULT = './logs/'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
	"""
	Defines the ops to conduct an optimization step. You can set a learning
	rate scheduler or pick your favorite optimizer here. This set of operations
	should be applicable to both ConvNet() and Siamese() objects.

	Args:
		loss: scalar float Tensor, full loss = cross_entropy + reg_loss

	Returns:
		train_op: Ops for optimization.
	"""
	
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	
	return train_op

	
def feature_extraction():
	"""
	This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
	model with restored parameters. From then on you can basically use that model in
	any way you want, for instance, feature extraction, finetuning or as a submodule
	of a larger architecture. However, this method should extract features from a
	specified layer and store them in data files such as '.h5', '.npy'/'.npz'
	depending on your preference. You will use those files later in the assignment.

	Args:
		[optional]
	Returns:
		None
	"""

	# ==================== CNN ======================
	
	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
	
	xs_train, ys_train = cifar10_dataset.train.next_batch(20000)
	#xs_test, ys_test = cifar10_dataset.test.images, cifar10_dataset.test.labels
	
	# Define session
	sess = tf.InteractiveSession()
	
	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')
		
	# Define and initialize network	
	net = ConvNet(10)
	
	# Calculate predictions
	logits = net.inference(x)	
	loss = net.loss(logits, y_)
	accuracy = net.accuracy(logits, y_)		
	
	# Get feature activation
	flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/Flatten/Activation:0")
	fc1 = tf.get_default_graph().get_tensor_by_name("ConvNet/FullLayer1/Activation:0")
	fc2 = tf.get_default_graph().get_tensor_by_name("ConvNet/FullLayer2/Activation:0")

	# Create a saver.
	saver = tf.train.Saver(tf.all_variables())
	
	# Init variables
	tf.initialize_all_variables().run()	
	
	# Restore variables from checkpoint
	checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
	saver.restore(sess, checkpoint_path)	
	print("Model loaded.")	
	
	# Run inference
	merged = tf.merge_all_summaries()	
	
	_, acc, flatten_np_train, fc1_np_train, fc2_np_train = sess.run([merged, accuracy, flatten, fc1, fc2], {x:xs_train, y_:ys_train})
#	_, acc, flatten_np_test, fc1_np_test, fc2_np_test = sess.run([merged, accuracy, flatten, fc1, fc2], {x:xs_test, y_:ys_test})		
	
	labels_train = np.zeros(ys_train.shape[0])
	i=0
	for label in ys_train:
		labels_train[i] = np.where(label==1)[0]
		i=i+1
	
#	labels_test = np.zeros(ys_test.shape[0])
#	i=0
#	for label in ys_test:
#		labels_test[i] = np.where(label==1)[0]
#		i=i+1	
	
	# Save features and labels
	np.save("./features/flatten_train.npy", flatten_np_train)
	np.save("./features/fc1_train.npy", fc1_np_train)
	np.save("./features/fc2_train.npy", fc2_np_train)
	np.save("./features/labels_train.npy", labels_train)

#	np.save("./features/flatten_test.npy", flatten_np_test)
#	np.save("./features/fc1_test.npy", fc1_np_test)
#	np.save("./features/fc2_test.npy", fc2_np_test)
#	np.save("./features/labels_test.npy", labels_test)
	

def initialize_folders():
	"""
	Initializes all folders in FLAGS variable.
	"""

	if not tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.MakeDirs(FLAGS.log_dir)

	if not tf.gfile.Exists(FLAGS.data_dir):
		tf.gfile.MakeDirs(FLAGS.data_dir)

	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
	"""
	Prints all entries in FLAGS variable.
	"""
	for key, value in vars(FLAGS).items():
		print(key + ' : ' + str(value))

def main(_):
	print_flags()

	initialize_folders()
	train()
	
if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
						help='Learning rate')
	parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
						help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
						help='Batch size to run trainer.')
	parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
						help='Frequency of evaluation on the train set')
	parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
						help='Frequency of evaluation on the test set')
	parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
						help='Frequency with which the model state is saved.')
	parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
						help='Directory for storing input data')
	parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
						help='Summaries log directory')
	parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
						help='Checkpoint directory')

	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run()
