from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
from train_cifar import train_cifar
from train_model import train_ctnet

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
	else:
		shutil.rmtree(FLAGS.log_dir, ignore_errors=True)
		tf.gfile.MakeDirs(FLAGS.log_dir)

	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
	else:
		shutil.rmtree(FLAGS.checkpoint_dir, ignore_errors=True)
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


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
FLAGS.weight_decay = 1e-4
FLAGS.nesterov_momentum = 0.9
# FLAGS.learning_rate = 0.001
# FLAGS.max_epochs = 60
# FLAGS.batch_size = 64
FLAGS.pretraining = False
FLAGS.xvalidation_folds = 10
# FLAGS.reduce_lr_epoch_1 = 50
# FLAGS.reduce_lr_epoch_2 = 50

# Exp-Gradient
FLAGS.alpha = 0.1

# Data
# FLAGS.normalization = False
# FLAGS.trainingpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_SKULLSTRIPPED_RESAMPLED/training_points.npy'
# FLAGS.testpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_SKULLSTRIPPED_RESAMPLED/test_points.npy'
# FLAGS.trainingpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_CTA/training_points.npy'
# FLAGS.testpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_CTA/test_points.npy'

# General CNN
FLAGS.batch_normalization = True
FLAGS.kernels = '11,5,5'
FLAGS.strides = '3,2,3'
FLAGS.maps = '32,32,2'
FLAGS.bases3d = False

# DenseNet
# FLAGS.growth_rate = 8
# FLAGS.depth = 40
FLAGS.total_blocks = 3
# FLAGS.keep_prob = 0.3
FLAGS.reduction = 1.0
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = False
# FLAGS.init_kernel = 11
# FLAGS.comp_kernel = 5

# RFNN
# FLAGS.sigmas = '1.5, 1.0, 0.5'


# Logging
FLAGS.print_freq = 1
FLAGS.eval_freq = 1
FLAGS.checkpoint_freq = 0
# FLAGS.log_dir = '/home/nicolab/DATA/logs/NCCT_CTA/DenseNet/' \
#     + str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
#     + str(FLAGS.learning_rate) + '_' \
#     + str(FLAGS.growth_rate) + '_' \
#     + str(FLAGS.depth) + '_' \
#     + str(FLAGS.total_blocks) + '_' \
#     + str(FLAGS.keep_prob) + '_' \
#     + str(FLAGS.reduction)

#    + str(FLAGS.reduce_lr_epoch_1) + '_' \


configs = [
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '0.5,0.5', 3, 3],
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '0.5,1.0', 3, 3],
#	[7, 3, 0.1, 8, 0.8, 25, 30, 128, 50, 50, '1.0,0.5', 3, 3],
#	[7, 3, 0.1, 8, 0.8, 25, 30, 64, 50, 50, '1.0,0.5', 3, 3],
#	[7, 3, 0.1, 8, 0.8, 25, 30, 128, 50, 50, '1.5,1.0,0.5', 3, 3],
	[7, 3, 0.01, 8, 0.8, 25, 30, 64, 50, 50, '1.5,1.0,0.5', 3, 3],
	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 50, 50, '1.5,1.0,0.5', 3, 3]
#	[7, 3, 0.001, 8, 0.8, 25, 30, 128, 50, 50, '1.0,0.5', 3, 3]

#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '1.0,1.0', 3, 3],
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '1.5,0.5', 3, 3],
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '0.5,1.5', 3, 3],

#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '1.5,1.0', 3, 3],
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '1.0,1.5', 3, 3],
#	[7, 3, 0.01, 8, 0.8, 25, 30, 128, 15, 50, '1.5,1.5', 3, 3]
]

for config in configs:
	FLAGS.init_kernel = config[0]
	FLAGS.comp_kernel = config[1]
	FLAGS.learning_rate = config[2]
	FLAGS.growth_rate = config[3]
	FLAGS.keep_prob = config[4]
	FLAGS.depth = config[5]
	FLAGS.max_epochs = config[6]
	FLAGS.batch_size = config[7]
	FLAGS.reduce_lr_epoch_1 = config[8]
	FLAGS.reduce_lr_epoch_2 = config[9]
	FLAGS.sigmas = config[10]
	FLAGS.init_order = config[11]
	FLAGS.comp_order = config[12]
	FLAGS.tol_var = 0.01
	FLAGS.t_max = 10

	#    FLAGS.log_dir = '/home/nicolab/DATA/logs/NCCT_CTA/DenseNet/bests/'
	FLAGS.log_dir = r'D:\Adam Hilbert\CT_Classification\code\Training\logs\rfnn\scale_sharing\\' \
					+ str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
					+ str(FLAGS.learning_rate) + '_' \
					+ str(FLAGS.growth_rate) + '_' \
					+ str(FLAGS.depth) + '_' \
					+ str(FLAGS.total_blocks) + '_' \
					+ str(FLAGS.keep_prob) + '_' \
					+ str(FLAGS.batch_size) + '_' \
					+ str(FLAGS.reduction) + '_' \
					+ str(FLAGS.sigmas) + '_' \
					+ str(FLAGS.init_order) + '_' \
					+ str(FLAGS.comp_order) + '_' \
					+ str(FLAGS.tol_var) + '_' \
					+ str(FLAGS.t_max) + '_max2'
	#    FLAGS.checkpoint_dir = '/home/nicolab/DATA/checkpoints/NCCT_CTA/DenseNet/' \
	FLAGS.checkpoint_dir = 'D:\Adam Hilbert\CT_Classification\code\Training\checkpoints\\' \
						   + str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
						   + str(FLAGS.learning_rate) + '_' \
						   + str(FLAGS.growth_rate) + '_' \
						   + str(FLAGS.depth) + '_' \
						   + str(FLAGS.total_blocks) + '_' \
						   + str(FLAGS.keep_prob) + '_' \
						   + str(FLAGS.reduction)

	initialize_folders()
	#    train_ctnet()
	train_cifar(FLAGS, NUM_GPUS)
