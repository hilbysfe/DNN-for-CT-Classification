from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import numpy as np
#from sklearn.metrics import auc
import math
import shutil
import pickle


from Utils import utils
from Models.rfnn import RFNN

LEARNING_RATE_DEFAULT = 0.05
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 70
EVAL_FREQ_DEFAULT = 1
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 2
SIGMAS_DEFAULT = "1.5,1.0,0.5"
KERNELS_DEFAULT = "11,3,3"
BASES_DEFAULT = "10,6,6"
MAPS_DEFAULT = "64,64,64"
MAXPOOLS_DEFAULT = "3,3,3,3"
L2 = 0.0005
HDROP = 0.0
CDROP = 0.0
DATAPATH = 'Normalized_Resampled_128x128x30'
MODEL_DEFAULT = 'RFNN_2d'

CHECKPOINT_DIR_DEFAULT = './checkpoints'
LOG_DIR_DEFAULT = './logs/'


def get_kernels():
	kernel = tf.get_default_graph().get_tensor_by_name("ConvLayer1/weights:0")
	alphas = tf.get_default_graph().get_tensor_by_name("L1_alphas:0")
	# print(kernel.get_shape())
	kernel_avg = tf.reduce_mean(kernel, axis=2)
	x_min = tf.reduce_min(kernel_avg)
	x_max = tf.reduce_max(kernel_avg)
	kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)

	# to tf.image_summary format [batch_size, height, width, channels]
	kernel_transposed = tf.transpose(kernel_avg, [2, 0, 1])
	# print(kernel_transposed.get_shape())

	return alphas, kernel_transposed


def accuracy_function(logits, labels):
	softmax = tf.nn.softmax(logits)
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
	tf.summary.scalar('Accuracy', accuracy)

	return accuracy, correct_prediction, softmax


def loss_function(logits, labels):
	with tf.variable_scope('Losses') as scope:
		with tf.name_scope('Cross_Entropy_Loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
			cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

			tf.add_to_collection('losses', cross_entropy_mean)
			tf.summary.scalar('cross_entropy', cross_entropy_mean)
		with tf.name_scope('Regularization_Loss'):
			reg_loss = tf.cast(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss'), tf.float16)

			tf.add_to_collection('losses', reg_loss)
			tf.summary.scalar('reg_loss', reg_loss)
		with tf.name_scope('Total_Loss'):
			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

			tf.summary.scalar('total_loss', loss)

	return loss


def train_step(loss, global_step):
	# decay_steps = 5*20
	# LEARNING_RATE_DECAY_FACTOR = 0.5

	# # Decay the learning rate exponentially based on the number of steps.
	# lr = tf.train.exponential_decay(FLAGS.learning_rate,
	# global_step,
	# decay_steps,
	# LEARNING_RATE_DECAY_FACTOR,
	# staircase=True)
	# tf.summary.scalar('learning_rate', lr)
	# train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1e-4).minimize(loss)
	train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
	return train_op


def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	global_step = tf.contrib.framework.get_or_create_global_step()

	# ====== DEFINE FEED_DICTIONARY ======
	def feed_dict(flag):
		if flag == 0:
			xs, ys = dataset.Training.next_batch(FLAGS.batch_size, bases3d=FLAGS.bases3d)
		else:
			if flag == 1:
				xs, ys = dataset.Validation.next_batch(FLAGS.batch_size, bases3d=FLAGS.bases3d)
			else:
				xs, ys = dataset.Test.next_batch(dataset.Test.num_examples, bases3d=FLAGS.bases3d)
		return {x: xs, y_: ys, is_training: flag == 0}

	# ====== LOAD DATASET ======
	print('Loading Dataset...')
	with open(FLAGS.trainingpath, 'rb') as handle:
	    training_points = pickle.load(handle)
	with open(FLAGS.testpath, 'rb') as handle:
	    test_points = pickle.load(handle)

	dataset = utils.DataSet(np.array(list(training_points.keys())), np.array(list(training_points.values())),
				np.array(list(test_points.keys())), np.array(list(test_points.values())),
				cross_validation_folds=10,
				normalize = FLAGS.normalization)
	print('Loading Dataset...done.')

	# ====== DEFINE SPACEHOLDERS ======
	with tf.name_scope('input'):
		if FLAGS.bases3d:
			x = tf.placeholder(tf.float16, [None, 512, 512, 30, 1], name='x-input')
		else:
			x = tf.placeholder(tf.float16, [None, 512, 512, 30], name='x-input')
		y_ = tf.placeholder(tf.float16, [None, 2], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')

	# ====== MODEL DEFINITION ======
	print('Defining model...')

	sigmas = [float(x) for x in FLAGS.sigmas.split(',')]
	kernels = [int(x) for x in FLAGS.kernels.split(',')]
	maps = [int(x) for x in FLAGS.maps.split(',')]
	bases = [int(x) for x in FLAGS.bases.split(',')]
	model = RFNN(
		n_classes=2,
		kernels=kernels,
		maps=maps,
		sigmas=sigmas,
		bases=bases,
		bases_3d=FLAGS.bases3d,
		is_training=is_training,
		batchnorm=FLAGS.batch_normalization
	)

	print('Defining model...done.')

	# ====== UNSUPERVISED PRE-TRAINING ======
	if FLAGS.pretraining:
		print('Pre-training model...')

		network_architecture = \
			{
				'Conv_kernels': kernels,
				'Conv_maps': maps
			}
		ae = Autoencoder(network_architecture)

		assign_ops, net = ae.load_weights(x, FLAGS.pretrained_weights_path, FLAGS.pretrained_biases_path, is_training)

		# Calculate predictions
		logits = model.inference(net)

		print('Pre-training model...done.')
	else:
		# Calculate predictions
		logits = model.inference(x)

	# ====== DEFINE LOSS, ACCURACY TENSORS ======
	print('Defining necessary OPs...')
	loss = loss_function(logits, y_)
	accuracy, prediction, scores = accuracy_function(logits, y_)

	# Call optimizer
	train_op = train_step(loss, global_step)

	# Create a saver.
	# saver = tf.train.Saver(tf.all_variables())

	# Merge all the summaries and write them out to log
	merged = tf.summary.merge_all()

	# Print initial kernels
	# alphas_tensor, kernels_tensor = get_kernels()	
	# alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])		
	# np.save('./Kernels/kernel_0.npy', kernels_array)
	# np.save('./Kernels/alphas_0.npy', alphas)

	print('Defining necessary OPs...done.')

	# Train
	training_steps = int(dataset.Training.num_examples / FLAGS.batch_size)
	# ====== DEFINE SESSION AND OPTIMIZE ======
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		print('Training model...')
		for f in range(FLAGS.xvalidation_folds):
			tf.global_variables_initializer().run()

			if FLAGS.pretraining:
				for assign_op in assign_ops:
					sess.run(assign_op)
					
			train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + str(f), sess.graph)
			test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test'  + str(f))

			max_acc = 0
			for i in range(int(FLAGS.max_epochs * training_steps)):
				# ------------ TRAIN -------------
				_ = sess.run([train_op], feed_dict=feed_dict(0))
				if i % (FLAGS.eval_freq * training_steps) == 0 or i == int(FLAGS.max_epochs * training_steps):
					# ------------ VALIDATON -------------
					tot_acc = 0.0
					tot_loss = 0.0
					steps = int(math.floor(dataset.Validation.num_examples/FLAGS.batch_size))
					for step in range(steps):
						acc_s, loss_s = sess.run([accuracy, loss], feed_dict=feed_dict(1))
						tot_acc += (acc_s / steps)
						tot_loss += (loss_s / steps)
							
					# Create a new Summary object with your measure
					summary = tf.Summary()
					summary.value.add(tag="Accuracy", simple_value=tot_acc)
					summary.value.add(tag="Loss", simple_value=tot_loss)
				
					test_writer.add_summary(summary, i)

					if tot_acc > max_acc:
						max_acc = tot_acc
					print('Validation accuracy at step %s: %s' % (i, tot_acc))

				if i % (FLAGS.print_freq * training_steps) == 0:
					# ------------ PRINT -------------
					summary = sess.run(merged, feed_dict=feed_dict(0))
					train_writer.add_summary(summary, i)

				# if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
				# checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
				# saver.save(sess, checkpoint_path, global_step=i)

			train_writer.close()
			test_writer.close()
			print('Max validation accuracy in fold %s: %s' % (f,max_acc))
			
			dataset.next_fold()



	# Print final kernels
	# alphas_tensor, kernels_tensor = get_kernels()	
	# alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])		
	# np.save('./Kernels/kernel_final.npy', kernels_array)
	# np.save('./Kernels/alphas_final.npy', alphas)

	# ======== ROC Analysis ==========
	# tresholds = 200
	# fpr_mean = np.zeros((tresholds))
	# tpr_mean = np.zeros((tresholds))
	# tr = np.linspace(0, 1, tresholds)
	# auc_list = []
	# iters = 20
	# test_acc = 0
	# for k in range(iters):

		# y_score, labels_test, acc = sess.run([scores, y_, accuracy], feed_dict=feed_dict(2))

		# # Compute ROC curve and ROC area for each class
		# fpr = np.zeros((tresholds))
		# tpr = np.zeros((tresholds))
		# for j in range(tresholds):
			# tp = 0
			# fp = 0
			# for i in range(np.shape(y_score)[0]):
				# if y_score[i][0] >= tr[j]:
					# if labels_test[i][0] == 1:
						# tp += 1
					# else:
						# fp += 1
			# tpr[j] += tp / np.sum(labels_test[:, 0] == 1)
			# fpr[j] += fp / np.sum(labels_test[:, 1] == 1)
		# auc_k = auc(fpr, tpr)
		# auc_list.append(auc_k)
		# tpr_mean = np.add(tpr_mean, np.divide(tpr, iters))
		# fpr_mean = np.add(fpr_mean, np.divide(fpr, iters))

		# test_acc += acc / iters

	# roc_auc = np.mean(np.array(auc_list))
	# std_auc = np.std(np.array(auc_list))

	# print('Acc/AUC/std : %s/%s/%s' % (test_acc, roc_auc, std_auc))


# if not os.path.isdir('./Statistics/' + FLAGS.model_name + '/3Dfinal/'):
# os.mkdir('./Statistics/' + FLAGS.model_name + '/3Dfinal/')
# if not os.path.isdir('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'):
# os.mkdir('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/')


# np.save('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'\
# + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
# + '_' + FLAGS.kernels.replace(",","_")  \
# + FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
# + '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
# + '_l2' + str(FLAGS.l2) \
# + '_tpr.npy', tpr_mean)
# np.save('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'\
# + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
# + '_' + FLAGS.kernels.replace(",","_")  \
# + FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
# + '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
# + '_l2' + str(FLAGS.l2) \
# + '_fpr.npy', fpr_mean)
# np.save('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'\
# + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
# + '_' + FLAGS.kernels.replace(",","_") \
# + FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
# + '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
# + '_l2' + str(FLAGS.l2) \
# + '_auc.npy', roc_auc)
# np.save('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'\
# + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
# + '_' + FLAGS.kernels.replace(",","_") \
# + FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
# + '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
# + '_l2' + str(FLAGS.l2) \
# + '_std.npy', std_auc)
# np.save('./Statistics/' + FLAGS.model_name + '/3Dfinal/' + FLAGS.dataset_name + '/'\
# + str(FLAGS.learning_rate) + '_' + str(FLAGS.batch_size) \
# + '_' + FLAGS.kernels.replace(",","_") \
# + FLAGS.maps.replace(",","_") + '_maxpool' + FLAGS.maxpool_kernels.replace(",","_") \
# + '_' + str(FLAGS.max_epochs) + 'epoch' + '_cdrop' + str(FLAGS.cdrop) + '_hdrop' + str(FLAGS.hdrop) \
# + '_l2' + str(FLAGS.l2) \
# + '_acc.npy', test_acc)


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


def str2bool(s):
	if s == "True":
		return True
	else:
		return False


def main(_):
#	print_flags()

	initialize_folders()
	train()


if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
						help='Learning rate')
	parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS_DEFAULT,
						help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
						help='Batch size to run trainer.')
	parser.add_argument('--pretraining', type=str2bool, default=False,
						help='Specify pretraining with Autoencoder')
	parser.add_argument('--xvalidation_folds', type=int, default=10,
						help='Specify number of cross-validation folds')
	parser.add_argument('--normalization', type=str2bool, default=False,
						help='Specify if normalization is applied to the data')
	parser.add_argument('--batch_normalization', type=str2bool, default=False,
						help='Specify if batch-normalization is used')
						
	parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
						help='Frequency of evaluation on the train set')
	parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
						help='Frequency of evaluation on the test set')
	parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
						help='Frequency with which the model state is saved.')
	parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
						help='Logging directory')

	parser.add_argument('--sigmas', type=str, default=SIGMAS_DEFAULT,
						help='Sigmas for RFNN')
	parser.add_argument('--kernels', type=str, default=KERNELS_DEFAULT,
						help='Kernel sizes of convolution')
	parser.add_argument('--maps', type=str, default=MAPS_DEFAULT,
						help='Amount of kernel maps of convolution')
	parser.add_argument('--bases', type=str, default=BASES_DEFAULT,
						help='Amount of basis functions to use')
	parser.add_argument('--bases3d', type=str2bool, default=True,
						help='Specify if 3d or 2d model is used')

	parser.add_argument('--hdrop', type=float, default=0.0,
						help='Hiddenlayer dropout')
	parser.add_argument('--cdrop', type=float, default=0.0,
						help='Convlayer dropout')
	parser.add_argument('--l2', type=float, default=0.0,
						help='Convlayer L2')

	parser.add_argument('--trainingpath', type=str,
						help='Path to file training_points has been saved to')
	parser.add_argument('--testpath', type=str,
						help='Path to file test_points has been saved to')
	parser.add_argument('--pretrained_weights_path', type=str, default="",
						help='Path to pretrained weights')
	parser.add_argument('--pretrained_biases_path', type=str, default="",
						help='Path to pretrained biases')

	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run()
