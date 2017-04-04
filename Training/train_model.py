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

from Utils import utils
from Models.rfnn import RFNN
from Models.alexnet import Alexnet
from Models.inception import Inception
from Models.c3d import C3D
from Models.alexnet import alexnet_v2_arg_scope
from Models.ctnet import CTNET

LEARNING_RATE_DEFAULT = 0.01
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 70
EVAL_FREQ_DEFAULT = 1
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 5
SIGMAS_DEFAULT = "0.5, 1, 1.5"
KERNELS_DEFAULT = "3,3,3,3"
MAPS_DEFAULT = "96,96"
MAXPOOLS_DEFAULT = "3,3,3,3"
L2 = 0.0
HDROP = 0.0
CDROP = 0.0
DATASET_NAME = 'Normalized_Resampled_128x128x30'

MODEL_DEFAULT = 'CTNET'

CHECKPOINT_DIR_DEFAULT = './checkpoints'

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

def train_step(loss):
	# global_step = tf.Variable(0, trainable=False)
	# rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 50, 0.99, staircase=False)
	# train_op = tf.train.GradientDescentOptimizer(rate).minimize(loss, global_step=global_step)
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	return train_op

def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	# np.random.seed(42)
		
	def feed_dict(flag):
		if flag==0:
			xs, ys = dataset.Training.next_batch(FLAGS.batch_size)
		else:
			if flag==1:
				xs, ys = dataset.Validation.next_batch(dataset.Validation.num_examples)
			else:
				xs, ys = dataset.Test.next_batch(dataset.Test.num_examples)
		return {x: xs, y_: ys, is_training: flag==0}	
	
	# Load data
	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + FLAGS.dataset_name
	
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
	if 'CTNET' in FLAGS.model_name:
		kernels = [float(x) for x in FLAGS.kernels.split(',')]
	if 'CTNET' in FLAGS.model_name:
		maps = [float(x) for x in FLAGS.maps.split(',')]
	if 'CTNET' in FLAGS.model_name:
		maxpools = [float(x) for x in FLAGS.maxpool_kernels.split(',')]
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
									),
				'CTNET'			: lambda: CTNET(
									n_classes=2,
									kernels = kernels,
									maps = maps,
									maxpool_kernels = maxpools,
									l2 = FLAGS.l2,
									is_training = is_training,
									dropout_rate_conv = FLAGS.cdrop,
									dropout_rate_hidden = FLAGS.hdrop
									)
			}[FLAGS.model_name]
	
	
	# Calculate predictions
	logits = model().inference(x)	
	
	loss = loss_function(logits, y_)
	accuracy, prediction, scores = accuracy_function(logits, y_)
	
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
	# print(training_steps)
	#print(validation_steps)
	
	max_acc = 0
	for i in range(int(FLAGS.max_epochs*training_steps)):
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(0))
		if i % (FLAGS.eval_freq*training_steps) == 0 or i==int(FLAGS.max_epochs*training_steps):				
			# ------------ VALIDATON -------------
			# tot_acc = 0.0
			# tot_loss = 0.0
			# steps = 4 #int(math.floor(dataset.Validation.num_examples/FLAGS.batch_size))
			# for step in range(steps):
				# acc_s, loss_s = sess.run([accuracy, loss], feed_dict=feed_dict(1))
				# tot_acc += (acc_s / steps)
				# tot_loss += (loss_s / steps)
						
			# # Create a new Summary object with your measure
			# summary = tf.Summary()
			# summary.value.add(tag="Accuracy", simple_value=tot_acc)
			# summary.value.add(tag="Loss", simple_value=tot_loss)

			summary, tot_acc = sess.run([merged, accuracy], feed_dict=feed_dict(1))
			
			# Add it to the Tensorboard summary writer
			test_writer.add_summary(summary, i)
			
			if tot_acc > max_acc:
				#checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
				#saver.save(sess, checkpoint_path)
				max_acc = tot_acc
			print('Validation accuracy at step %s: %s' % (i, tot_acc))
					
#			if i == FLAGS.max_steps:
#				np.savetxt("confusion_matrix.csv", conf.astype(int), delimiter=",")
								
		if i % FLAGS.print_freq == 0:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(0))
			train_writer.add_summary(summary[0], i)
		
		# if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
			# checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
			# saver.save(sess, checkpoint_path, global_step=i)
	train_writer.close()
	test_writer.close()
	print('Max accuracy : %s' % (max_acc))
	
	# --------- ROC Analysis ----------
	tresholds = 200
	fpr_mean = np.zeros((tresholds))
	tpr_mean = np.zeros((tresholds))
	tr = np.linspace(0,1,tresholds)
	auc_list = []
	iters = 20
	for k in range(iters):
		# steps = 4 #int(math.floor(dataset.Test.num_examples/FLAGS.batch_size))
		# y_score = np.zeros((int(dataset.Test.num_examples/steps)*steps,2))
		# labels_test = np.zeros((steps*int(dataset.Test.num_examples/steps),2))
		# for step in range(steps):
			# offset = step*int(dataset.Test.num_examples/steps)
		
			# y_score_s, labels_test_s = sess.run([scores, y_], feed_dict=feed_dict(2))		
		y_score, labels_test = sess.run([scores, y_], feed_dict=feed_dict(2))
			
			# y_score[offset:offset+int(dataset.Test.num_examples/steps),:] = y_score_s
			# labels_test[offset:offset+int(dataset.Test.num_examples/steps),:] = labels_test_s
			
		# print(y_score)
		# Compute ROC curve and ROC area for each class
		fpr = np.zeros((tresholds))
		tpr = np.zeros((tresholds))
		for j in range(tresholds):
			tp = 0
			fp = 0
			for i in range(np.shape(y_score)[0]):
				if y_score[i][0] >= tr[j]:
					if labels_test[i][0] == 1:
						tp += 1
					else:
						fp += 1
			tpr[j] += tp/np.sum(labels_test[:,0]==1)
			fpr[j] += fp/np.sum(labels_test[:,1]==1)
		auc_k = auc(fpr, tpr)
		auc_list.append(auc_k)
		tpr_mean = np.add(tpr_mean, np.divide(tpr, iters))
		fpr_mean = np.add(fpr_mean, np.divide(fpr, iters))
	# print(fpr)
	# print(tpr)
	roc_auc = np.mean(np.array(auc_list))
	std_auc = np.std(np.array(auc_list))
	
	if not os.path.isdir('./Statistics/' + FLAGS.model_name + '/'):
		os.mkdir('./Statistics/' + FLAGS.model_name + '/')
	if not os.path.isdir('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/'):
		os.mkdir('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/')
	
	np.save('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/'\
			+ str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
			+ '_' + FLAGS.kernels.replace(",","_")  \
			+ FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
			+ '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
			+ '_l2' + str(FLAGS.l2) \
			+ '_tpr.npy', tpr_mean)
	np.save('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/'\
			+ str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
			+ '_' + FLAGS.kernels.replace(",","_")  \
			+ FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
			+ '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
			+ '_l2' + str(FLAGS.l2) \
			+ '_fpr.npy', fpr_mean)
	np.save('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/'\
			+ str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
			+ '_' + FLAGS.kernels.replace(",","_") \
			+ FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
			+ '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
			+ '_l2' + str(FLAGS.l2) \
			+ '_auc.npy', roc_auc)		
	np.save('./Statistics/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/'\
			+ str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
			+ '_' + FLAGS.kernels.replace(",","_") \
			+ FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
			+ '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
			+ '_l2' + str(FLAGS.l2) \
			+ '_std.npy', std_auc)	
	print('AUC/std : %s/%s' % (roc_auc, std_auc))

def initialize_folders():
	"""
	Initializes all folders in FLAGS variable.
	"""
	if not tf.gfile.Exists(FLAGS.log_dir):
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
	parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
						help='Checkpoint directory')
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

	FLAGS.log_dir = './logs/' + FLAGS.model_name + '/' + FLAGS.dataset_name + '/' \
		+ '2D/128/4Layer/' + str(FLAGS.learning_rate) + \
		'_' + str(FLAGS.batch_size) + '_' + FLAGS.kernels.replace(",","_")  + '_'\
		 + FLAGS.maps.replace(",","_") + '_' + str(FLAGS.max_epochs) \
		+ 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_l2-' + str(FLAGS.l2) + '_hdrop' + str(FLAGS.hdrop) 
		#  + '_cdrop' + str(FLAGS.cdrop) + '_l2-' + str(FLAGS.l2) + '_hdrop' + str(FLAGS.hdrop) 
		# + '_mp' + FLAGS.maxpool_kernels.replace(",","_") \
		
	tf.app.run()
