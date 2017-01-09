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
BATCH_SIZE_DEFAULT = 1
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10

LOG_DIR_DEFAULT = './logs/'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
	
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	
	return train_op
	

def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	np.random.seed(42)

		
	def feed_dict(train):
		if train:
			xs, ys = Training_Set.next_batch(FLAGS.batch_size)
		else:
			xs, ys = Validation_Set.next_batch(Validation_Set.num_examples)
		return {x: xs, y_: ys, is_training: train}	
	
	# Load data
	data_dir = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\'
	images_root_sub = 'MRCLEAN_CT24h\\'
	image_sub = 'CT24h\\thick\\'

	label_root_sub = 'MRCLEAN\\'
	label_filename = 'MRCLEAN_MRSDICH.xlsx'

	Training_Set, Validation_Set = utils.read_datasets(data_dir, images_root_sub, image_sub, label_root_sub, label_filename, 0.3)
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 224, 224, 12], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')
	
	# Calculate predictions
	logits, _ = alexnet.alexnet_v2(eval_inputs, is_training=False)
	
	loss = alexnet.loss(logits, y_)
	accuracy = alexnet.accuracy(logits, y_)
		
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
							
		elif i % FLAGS.print_freq == 0:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(True))
			train_writer.add_summary(summary[0], i)
		
		if i % FLAGS.checkpoint_freq == 0 or i == FLAGS.max_steps:
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
