from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc
import math

import cifar10_utils
import shutil

from Utils import utils
from Models.rfnn import RFNN
from Models.alexnet import Alexnet
from Models.inception import Inception
from Models.c3d import C3D
from Models.alexnet import alexnet_v2_arg_scope
from Models.ctnet import CTNET

LEARNING_RATE_DEFAULT = 0.005
BATCH_SIZE_DEFAULT = 128
MAX_EPOCHS_DEFAULT = 70
EVAL_FREQ_DEFAULT = 3
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 3
SIGMAS_DEFAULT = "3.5, 1.5, 1.5"
KERNELS_DEFAULT = "11,3,3"
MAPS_DEFAULT = "64,64,64"
MAXPOOLS_DEFAULT = "3,3,3,3"
L2 = 0.0005
HDROP = 0.0
CDROP = 0.0
DATASET_NAME = 'Normalized_Resampled_128x128x30'
MODEL_DEFAULT = 'RFNN_2d'
LOG_DIR_DEFAULT = './logs/'


def get_kernels():	
	
	kernel0 = tf.get_default_graph().get_tensor_by_name("ConvLayer1/weights_0:0")
	kernel1 = tf.get_default_graph().get_tensor_by_name("ConvLayer1/weights_1:0")
	kernel2 = tf.get_default_graph().get_tensor_by_name("ConvLayer1/weights_2:0")
	
	alphas = tf.get_default_graph().get_tensor_by_name("L1_alphas:0")
	# alphas = tf.get_default_graph().get_tensor_by_name("ConvLayer2/weights:0")
	kernel_avg0 = tf.reduce_mean(kernel0, axis=2)
	kernel_avg1 = tf.reduce_mean(kernel1, axis=2)
	kernel_avg2 = tf.reduce_mean(kernel2, axis=2)
	
	# to tf.image_summary format [batch_size, height, width, channels]
	kernel_transposed0 = tf.transpose(kernel_avg0, [2, 0, 1])
	kernel_transposed1 = tf.transpose(kernel_avg1, [2, 0, 1])
	kernel_transposed2 = tf.transpose(kernel_avg2, [2, 0, 1])
		
	return alphas, tf.pack([kernel_transposed0, kernel_transposed1, kernel_transposed2])

def accuracy_function(logits, labels):	
	softmax = tf.nn.softmax(logits)
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('Accuracy', accuracy)

	return accuracy, correct_prediction, softmax

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

def train_step(loss, global_step):
	decay_steps = 390*50
	LEARNING_RATE_DECAY_FACTOR = 0.5

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
	tf.summary.scalar('learning_rate', lr)
	train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
	return train_op

def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	global_step = tf.contrib.framework.get_or_create_global_step()
	# np.random.seed(42)
		
	def feed_dict(train):
		if train:
			xs, ys = cifar10_dataset.train.next_batch(FLAGS.batch_size)
		else:
			xs, ys = cifar10_dataset.test.images, cifar10_dataset.test.labels
		return {x: xs, y_: ys, is_training: train}	
	
	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
	
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')

	
	# Model definition
	if 'RFNN' in FLAGS.model_name:
		sigmas = [float(x) for x in FLAGS.sigmas.split(',')]
	kernels = [int(x) for x in FLAGS.kernels.split(',')]
	if 'CTNET' in FLAGS.model_name:
		maps = [int(x) for x in FLAGS.maps.split(',')]
	if 'CTNET' in FLAGS.model_name:
		maxpools = [int(x) for x in FLAGS.maxpool_kernels.split(',')]
	model = {
				'RFNN_2d' 		: lambda: RFNN(
									n_classes = 10,
									kernels=kernels,
									is_training = is_training,
									sigmas=sigmas,
									bases_3d = False
									),
				'CTNET'			: lambda: CTNET(
									n_classes=10,
									kernels = kernels,
									maps = maps,
									maxpool_kernels = maxpools,
									l2 = FLAGS.l2,
									is_training = is_training,
									dropout_rate_conv = FLAGS.cdrop,
									dropout_rate_hidden = FLAGS.hdrop,
									conv3d = False
									)
			}[FLAGS.model_name]
	
	
	# Calculate predictions
	logits = model().inference(x)	
	
	loss = loss_function(logits, y_)
	accuracy, prediction, scores = accuracy_function(logits, y_)

	
	# Call optimizer
	train_op = train_step(loss, global_step)
		
	# Merge all the summaries and write them out to log
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',	sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	
	tf.global_variables_initializer().run()
	
	# Print initial kernels
	alphas_tensor, kernels_tensor = get_kernels()	
	alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])		
	np.save('./Kernels/kernel_0.npy', kernels_array)
	np.save('./Kernels/alphas_0.npy', alphas)
	
	# Train
	max_acc = 0
	training_steps = int(cifar10_dataset.train.num_examples/FLAGS.batch_size)
	
	for i in range(int(FLAGS.max_epochs*training_steps)):
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(True))
		if i % (FLAGS.eval_freq*training_steps) == 0:				
			# ------------ VALIDATON -------------
			summary, tot_acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))			
			test_writer.add_summary(summary, i)
			
			if tot_acc > max_acc:
				max_acc = tot_acc
			print('Validation accuracy at step %s: %s' % (i, tot_acc))
		
		if i % (FLAGS.print_freq*training_steps) == 0:
			# ------------ PRINT -------------
			summary = sess.run(merged, feed_dict=feed_dict(True))
			train_writer.add_summary(summary, i)

	print('Max accuracy : %s' % (max_acc))
	train_writer.close()
	test_writer.close()
	
	# Print final kernels
	alphas_tensor, kernels_tensor = get_kernels()	
	alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])		
	np.save('./Kernels/kernel_final.npy', kernels_array)
	np.save('./Kernels/alphas_final.npy', alphas)
	
		

def initialize_folders():
	"""
	Initializes all folders in FLAGS variable.
	"""
	if not tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.MakeDirs(FLAGS.log_dir)
	else:
		shutil.rmtree(FLAGS.log_dir)
		tf.gfile.MakeDirs(FLAGS.log_dir)

	# if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		# tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

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
	parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
						help='Logging directory')
	parser.add_argument('--model_name', type = str, default = MODEL_DEFAULT,
						help='Model name')
	parser.add_argument('--sigmas', type = str, default = SIGMAS_DEFAULT,
						help='Sigmas for RFNN')
	parser.add_argument('--kernels', type = str, default = KERNELS_DEFAULT,
						help='Kernel sizes of convolution')									
	parser.add_argument('--maps', type = str, default = MAPS_DEFAULT,
						help='Amount of kernel maps of convolution')	
	parser.add_argument('--maxpool_kernels', type = str, default = MAXPOOLS_DEFAULT,
						help='Kernelsize of maxpool layers')			
	parser.add_argument('--hdrop', type = float, default = HDROP,
						help='Hiddenlayer dropout')	
	parser.add_argument('--cdrop', type = float, default = CDROP,
						help='Convlayer dropout')		
	parser.add_argument('--l2', type = float, default = L2,
						help='Convlayer L2')		
	parser.add_argument('--dataset_name', type = str, default = DATASET_NAME,
						help='Name of the dataset')									
						
						
	FLAGS, unparsed = parser.parse_known_args()

	# FLAGS.log_dir = './logs/test/'
		# + FLAGS.model_name + '/final/' + FLAGS.dataset_name + '/' \
		# + '3D/128/' + str(FLAGS.learning_rate) + \
		# '_' + str(FLAGS.batch_size) + '_' + FLAGS.kernels.replace(",","_")  + '_'\
		 # + FLAGS.maps.replace(",","_") + '_' + str(FLAGS.max_epochs) \
		# + 'epoch'  + '_mp' + FLAGS.maxpool_kernels.replace(",","_") + '_l2-' + str(FLAGS.l2) + '_hdrop' + str(FLAGS.hdrop)
		
		#  + '_cdrop' + str(FLAGS.cdrop) + '_l2-' + str(FLAGS.l2) + '_hdrop' + str(FLAGS.hdrop) 
		# + '_mp' + FLAGS.maxpool_kernels.replace(",","_") \
		
	tf.app.run()
