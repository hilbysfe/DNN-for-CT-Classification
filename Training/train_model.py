from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import argparse
import os
import tensorflow as tf
import numpy as np

import cifar10_utils

import utils
from Models.rfnn import RFNN
from Models.alexnet import Alexnet
from Models.inception import Inception
from Models.c3d import C3D
from Models.alexnet import alexnet_v2_arg_scope

LEARNING_RATE_DEFAULT = 0.01
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 40
EVAL_FREQ_DEFAULT = 18
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 5
SIGMAS_DEFAULT = "0.5, 1, 1.5"

MODEL_DEFAULT = 'RFNN_2d'

CHECKPOINT_DIR_DEFAULT = './checkpoints'

def accuracy_function(logits, labels):	
	softmax = tf.nn.softmax(logits)
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('Accuracy', accuracy)

	return accuracy

def loss_function(logits, labels):	
	with tf.variable_scope('Losses') as scope:		
		with tf.name_scope('Cross_Entropy_Loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
			cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
			
			tf.add_to_collection('losses', cross_entropy_mean)		
			tf.summary.scalar('cross_entropy', cross_entropy_mean)
		with tf.name_scope('Regularization_Loss'):
			reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
			
			tf.add_to_collection('losses', reg_loss)
			tf.summary.scalar('reg_loss', reg_loss)
		with tf.name_scope('Total_Loss'):
			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
			
			tf.summary.scalar('total_loss', loss)

	return loss	

def train_step(loss):	
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	return train_op

def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	# np.random.seed(42)
		
	def feed_dict(train):
		if train:
			xs, ys = dataset.Training.next_batch(FLAGS.batch_size)
		else:
			xs, ys = dataset.Validation.next_batch(int(dataset.Validation.num_examples/2))
		return {x: xs, y_: ys, is_training: train}	
	
	# Load data
	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + 'Normalized_RigidAligned_128x128x30+Flipped'
	
	label_filename = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\MRCLEAN\\MRCLEAN_MRSDICH.xlsx'

	dataset = utils.read_dataset(image_dir, label_filename)
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 128, 128, 16], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')

	
	
	# Model definition
	if 'RFNN' in FLAGS.model_name:
		sigmas = [float(x) for x in FLAGS.sigmas.split(',')]
	model = {
				'RFNN_2d' 		: lambda: RFNN(
									n_classes = 2,
									is_training = is_training,
									sigmas=sigmas,
									bases_3d = False
									),
				'RFNN_3d' 		: lambda: RFNN(
									n_classes = 2,
									is_training = is_training,
									sigmas=sigmas,
									bases_3d = True
									),
				'alexnet_2d'	: lambda: Alexnet(
									kernels_3d=False,
									num_classes=2,
									is_training=is_training,
									scope=alexnet.alexnet_v2_arg_scope()
									),
				'alexnet_3d' 	: lambda: Alexnet(
									kernels_3d=True,
									num_classes=2,
									is_training=is_training
									),
				'c3d'			: lambda: C3D(
									num_classes=2,
									is_training=is_training,
									dropout_keep_prob=0.5
									),
				'Inception'		: lambda: Inception(
									num_classes=2,
									is_training=is_training
									)
			}[FLAGS.model_name]
	
	
	# Calculate predictions
	logits = model().inference(x)	
	
	loss = loss_function(logits, y_)
	accuracy = accuracy_function(logits, y_)
	
	# Call optimizer
	train_op = train_step(loss)
	
	# Create a saver.
#	saver = tf.train.Saver(tf.all_variables())
	
	# Merge all the summaries and write them out to log
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',	sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	tf.global_variables_initializer().run()
	
	# Train
	max_acc = 0
	training_steps = int(dataset.Training.num_examples/FLAGS.batch_size)
	validation_steps = int(dataset.Validation.num_examples/FLAGS.batch_size)
	#print(training_steps)
	#print(validation_steps)
	
	max_acc = 0
	for i in range(int(FLAGS.max_epochs*training_steps)):
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(True))
		if i % FLAGS.eval_freq == 0 or i==int(FLAGS.max_epochs*training_steps):				
			# ------------ TEST -------------
			acc1, _ = sess.run([accuracy, loss], feed_dict=feed_dict(False))
			acc2, _ = sess.run([accuracy, loss], feed_dict=feed_dict(False))
			
			acc = (acc1+acc2)/2
			# loss = (loss1+loss2)/2
			# Create a new Summary object with your measure
			summary = tf.Summary()
			summary.value.add(tag="Accuracy", simple_value=acc)
			# summary.value.add(tag="Loss", simple_value=loss)

			# Add it to the Tensorboard summary writer
			test_writer.add_summary(summary, i)
			
			if acc > max_acc:
				#checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
				#saver.save(sess, checkpoint_path)
				max_acc = acc
			print('Validation accuracy at step %s: %s' % (i, acc))
					
#			if i == FLAGS.max_steps:
#				np.savetxt("confusion_matrix.csv", conf.astype(int), delimiter=",")
								
		elif i % FLAGS.print_freq == 0:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(True))
			train_writer.add_summary(summary[0], i)
		
		# if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
			# checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
			# saver.save(sess, checkpoint_path, global_step=i)
	train_writer.close()
	test_writer.close()
	print('Max accuracy : %s' % (max_acc))


def initialize_folders():
	"""
	Initializes all folders in FLAGS variable.
	"""
	if not tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.MakeDirs(FLAGS.log_dir)

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
	parser.add_argument('--max_epochs', type = int, default = MAX_EPOCHS_DEFAULT,
						help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
						help='Batch size to run trainer.')
	parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
						help='Frequency of evaluation on the train set')
	parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
						help='Frequency of evaluation on the test set')
	parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
						help='Frequency with which the model state is saved.')
	parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
						help='Checkpoint directory')
	parser.add_argument('--model_name', type = str, default = MODEL_DEFAULT,
						help='Model name')
	parser.add_argument('--sigmas', type = str, default = SIGMAS_DEFAULT,
						help='Sigmas for RFNN')
						
	FLAGS, unparsed = parser.parse_known_args()

	FLAGS.log_dir = './logs/' + FLAGS.model_name + '/' \
		+ 'NormalizedRigid/1Layer/' + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) + '_' + FLAGS.sigmas.replace(",","_") + '_10bases_norm_maxpool/'
	
	tf.app.run()
