from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
from train_cifar import train_cifar

NUM_GPUS = 1

class FLAGS():
	def __init__(self):
		return

def initialize_folders():
	"""
    Initializes all folders in FLAGS variable.
    """
	if not tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.MakeDirs(FLAGS.log_dir)
#	else:
#		shutil.rmtree(FLAGS.log_dir, ignore_errors=True)
#		tf.gfile.MakeDirs(FLAGS.log_dir)

#	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
#		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
#	else:
#		shutil.rmtree(FLAGS.checkpoint_dir, ignore_errors=True)
#		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

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

FLAGS = FLAGS()

# Training
FLAGS.pretraining = False
FLAGS.max_epochs = 60
FLAGS.batch_size = 64
FLAGS.xvalidation_folds = 4
FLAGS.trials = 3
FLAGS.nesterov_momentum = 0.9
FLAGS.learning_rate = 0.1
FLAGS.weight_decay = 1e-4

# LSUV
FLAGS.t_max = 10
FLAGS.t_min = 5
FLAGS.tol_var = 0.01

# Data
FLAGS.cifar_path = r'D:\Adam Hilbert\CT_Classification\code\Training\cifar10\cifar-10-batches-py'
FLAGS.normalization = True
FLAGS.X_dim = 32

# General CNN
FLAGS.bases3d = False
FLAGS.init_kernel = 3
FLAGS.comp_kernel = 3
FLAGS.keep_prob = 0.8
FLAGS.init_order = 3
FLAGS.comp_order = 3

# DenseNet
FLAGS.total_blocks = 3
FLAGS.reduction = 0.5
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = True

# Logging
FLAGS.print_freq = 3
FLAGS.eval_freq = 1

configs = [
#	DEEP
	[12, 40, 45, 200, '1.0,0.5', '1.0,0.5', "cnn", '45, 45', 10000, 60],
	[12, 40, 30, 55, '1.0,0.5', '1.0,0.5', "cnn", '45, 45', 5000, 75],
	[12, 40, 30, 55, '1.0,0.5', '1.0,0.5', "cnn", '45, 45', 1000, 75]
]


for config in configs:

	FLAGS.growth_rate = config[0]
	FLAGS.depth = config[1]
	FLAGS.reduce_lr_epoch_1 = config[2]
	FLAGS.reduce_lr_epoch_2 = config[3]
	FLAGS.init_sigmas = config[4]
	FLAGS.comp_sigmas = config[5]
	FLAGS.rfnn = config[6]
	FLAGS.thetas = config[7]
	FLAGS.training_size = config[8]
	FLAGS.max_epochs = config[9]

	FLAGS.log_dir = r'D:\Experiments\logs\CIFAR' + str(int(FLAGS.training_size/1000)) + 'K\\' \
					+ str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
					+ str(FLAGS.learning_rate) + '_' \
					+ str(FLAGS.growth_rate) + '_' \
					+ str(FLAGS.depth) + '_' \
					+ str(FLAGS.total_blocks) + '_' \
					+ str(FLAGS.keep_prob) + '_' \
					+ str(FLAGS.batch_size) + '_' \
					+ str(FLAGS.rfnn) + '_' \
					+ str(FLAGS.init_sigmas) + str(FLAGS.comp_sigmas) + '_' \
					+ str(FLAGS.reduce_lr_epoch_1) + '_' \
					+ str(FLAGS.reduce_lr_epoch_2) + '_' + str(FLAGS.max_epochs) + '_' \
					+ str(FLAGS.reduction) + 'adaptWD' \
					+ str(FLAGS.thetas)

	FLAGS.checkpoint_dir = r'D:\Experiments\checkpoints\CIFAR' + str(int(FLAGS.training_size/1000)) + 'K\\' \
					   + str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
					   + str(FLAGS.learning_rate) + '_' \
					   + str(FLAGS.growth_rate) + '_' \
					   + str(FLAGS.depth) + '_' \
					   + str(FLAGS.total_blocks) + '_' \
					   + str(FLAGS.keep_prob) + '_' \
					   + str(FLAGS.batch_size) + '_' \
					   + str(FLAGS.rfnn) + '_' \
					   + str(FLAGS.init_sigmas) + str(FLAGS.comp_sigmas) + '_' \
					   + str(FLAGS.reduce_lr_epoch_1) + '_' \
					   + str(FLAGS.reduce_lr_epoch_2) + '_' + str(FLAGS.max_epochs) + '_' \
					   + str(FLAGS.reduction) + 'adaptWD' \
					   + str(FLAGS.thetas)

	FLAGS.stat_dir = r'D:\Experiments\stats\CIFAR' + str(int(FLAGS.training_size/1000)) + 'K\\' \
					 + str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
					 + str(FLAGS.learning_rate) + '_' \
					 + str(FLAGS.growth_rate) + '_' \
					 + str(FLAGS.depth) + '_' \
					 + str(FLAGS.total_blocks) + '_' \
					 + str(FLAGS.keep_prob) + '_' \
					 + str(FLAGS.batch_size) + '_' \
					 + str(FLAGS.rfnn) + '_' \
					 + str(FLAGS.init_sigmas) + str(FLAGS.comp_sigmas) + '_' \
					 + str(FLAGS.reduce_lr_epoch_1) + '_' \
					 + str(FLAGS.reduce_lr_epoch_2) + '_' + str(FLAGS.max_epochs) + '_' \
					 + str(FLAGS.reduction) + 'adaptWD' \
					 + str(FLAGS.thetas)

	initialize_folders()
	train_cifar(FLAGS, NUM_GPUS)
