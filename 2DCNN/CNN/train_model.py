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

	

def train():
	"""
	Performs training and evaluation of ConvNet model.

	First define your graph using class ConvNet and its methods. Then define
	necessary operations such as trainer (train_step in this case), savers
	and summarizers. Finally, initialize your model within a tf.Session and
	do the training.

	---------------------------
	How to evaluate your model:
	---------------------------
	Evaluation on test set should be conducted over full batch, i.e. 10k images,
	while it is alright to do it over minibatch for train set.

	---------------------------------
	How often to evaluate your model:
	---------------------------------
	- on training set every print_freq iterations
	- on test set every eval_freq iterations

	------------------------
	Additional requirements:
	------------------------
	Also you are supposed to take snapshots of your model state (i.e. graph,
	weights and etc.) every checkpoint_freq iterations. For this, you should
	study TensorFlow's tf.train.Saver class. For more information, please
	checkout:
	[https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
	"""

	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	np.random.seed(42)

		
	def feed_dict(train):
		if train:
			xs, ys = cifar10_dataset.train.next_batch(FLAGS.batch_size)
		else:
			xs, ys = cifar10_dataset.test.images, cifar10_dataset.test.labels
		return {x: xs, y_: ys, is_training: train}	
	
	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')
		
	# Define and initialize network	
	net = ConvNet(10, is_training)
	
	# Calculate predictions
	logits = net.inference(x)	
	loss = net.loss(logits, y_)
	accuracy = net.accuracy(logits, y_)
	# confusion = net.confusion(logits, y_)
		
	# Call optimizer
	train_op = train_step(loss)
	
	# Create a saver.
	saver = tf.train.Saver(tf.all_variables())
	
	# Merge all the summaries and write them out to log
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',	sess.graph)
	test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
	tf.initialize_all_variables().run()
	
	# Train
	max_acc = 0
	for i in range(1, 1+FLAGS.max_steps):
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(True))
		if i % FLAGS.eval_freq == 0 or i == 1:				
			# ------------ TEST -------------
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			if acc > max_acc:
				#checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
				#saver.save(sess, checkpoint_path)
				max_acc = acc
			print('Accuracy at step %s: %s' % (i, acc))
					
#			if i == FLAGS.max_steps:
#				np.savetxt("confusion_matrix.csv", conf.astype(int), delimiter=",")
						
		
#			print('Adding run metadata for', i)
		elif i % FLAGS.print_freq == 0:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(True))
			train_writer.add_summary(summary[0], i)
		
		if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
			checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=i)
	train_writer.close()
	test_writer.close()
	print('Max accuracy : %s' % (max_acc))


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
