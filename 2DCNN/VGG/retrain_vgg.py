from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from vgg_convnet import VGGConvNet
import vgg as vgg

import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
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
	########################
	# PUT YOUR CODE HERE  #
 	########################
	
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	
	########################
	# END OF YOUR CODE	#
	########################

	return train_op

def train():
	"""
	Performs training and evaluation of your model.

	First define your graph using vgg.py with your fully connected layer.
	Then define necessary operations such as trainer (train_step in this case),
	savers and summarizers. Finally, initialize your model within a
	tf.Session and do the training.

	---------------------------------
	How often to evaluate your model:
	---------------------------------
	- on training set every PRINT_FREQ iterations
	- on test set every EVAL_FREQ iterations

	---------------------------
	How to evaluate your model:
	---------------------------
	Evaluation on test set should be conducted over full batch, i.e. 10k images,
	while it is alright to do it over minibatch for train set.
	"""

	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	np.random.seed(42)

	########################
	# PUT YOUR CODE HERE  #
	########################
		
	def feed_dict(train, it):
		if train:
			xs, ys = cifar10_dataset.train.next_batch(FLAGS.batch_size)
		else:
			xs, ys = cifar10_dataset.test.images, cifar10_dataset.test.labels
		return {x: xs, y_: ys, is_training: train, is_refine: it > FLAGS.refine_after_k}  

	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, None, None, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		is_refine = tf.placeholder(tf.bool, name='is-refine')
		is_training = tf.placeholder(tf.bool, name='is-training')
		
	# Define and initialize network	
	net = VGGConvNet(10, is_training, is_refine)
	
	pool5, assign_ops = vgg.load_pretrained_VGG16_pool5(x)
	pool5 = tf.cond(is_refine, lambda: pool5, lambda: tf.stop_gradient(pool5))
	
	# Calculate predictions	
	logits = net.inference(pool5)
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
	
	# Init VGG
	for assign_op in assign_ops:
		sess.run(assign_op)
	
	# Train
	max_acc = 0
	for i in range(1, 1+FLAGS.max_steps):
		if i == FLAGS.refine_after_k:
			FLAGS.learning_rate = FLAGS.learning_rate * 10
	
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(True, i))
		if i % FLAGS.print_freq == 0 or i == 1:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(True, i))
			train_writer.add_summary(summary[0], i)
		if i % FLAGS.eval_freq == 0 or i == 1:  			
			# ------------ TEST -------------
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, i))
			test_writer.add_summary(summary, i)
			if acc > max_acc:
				#checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
				#saver.save(sess, checkpoint_path)
				max_acc = acc
			print('Accuracy at step %s: %s' % (i, acc))
					
#			if i == FLAGS.max_steps:
#				np.savetxt("confusion_matrix.csv", conf.astype(int), delimiter=",")
						
		
#			print('Adding run metadata for', i)		
		if i % FLAGS.checkpoint_freq == 0 or i == FLAGS.max_steps: # or i == FLAGS.max_steps:
			checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=i)
	train_writer.close()
	test_writer.close()
	print('Max accuracy : %s' % (max_acc))
	
	
	
	########################
	# END OF YOUR CODE	#
	########################

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
	parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
					  help='Number of steps after which to refine VGG model parameters (default 0).')
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
