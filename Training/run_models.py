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
FLAGS.weight_decay = 1e-4
FLAGS.nesterov_momentum = 0.9
# FLAGS.learning_rate = 0.001
# FLAGS.max_epochs = 60
# FLAGS.batch_size = 64
FLAGS.pretraining = False

# FLAGS.reduce_lr_epoch_1 = 50
# FLAGS.reduce_lr_epoch_2 = 50

# Exp-Gradient
#FLAGS.alpha = 0.1

# Data
#FLAGS.cifar_path = '/home/hilbysfe/dev/DNN-for-CT-Classification/Training/cifar-10-batches-py'
#FLAGS.cifar_path = r'D:\Adam Hilbert\CT_Classification\code\Training\cifar10\cifar-10-batches-py'
FLAGS.normalization = True
#FLAGS.trainingpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_SKULLSTRIPPED_RESAMPLED/training_points.npy'
#FLAGS.testpath = '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_SKULLSTRIPPED_RESAMPLED/test_points.npy'

FLAGS.trainingpath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D_affected_side\training_points.npy'
FLAGS.testpath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D_affected_side\test_points.npy'

#FLAGS.trainingpath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D\training_points.npy'
#FLAGS.testpath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D\test_points.npy'

#FLAGS.trainingpath = r'D:\Adam Hilbert\Data\data_binaries\NCCT\training_points.npy'
#FLAGS.testpath = r'D:\Adam Hilbert\Data\data_binaries\NCCT\test_points.npy'

FLAGS.label_attribute = 'collaterals'


# General CNN
#FLAGS.batch_normalization = True
#FLAGS.kernels = '11,5,5'
#FLAGS.strides = '3,2,3'
#FLAGS.maps = '32,32,2'
FLAGS.bases3d = False

# DenseNet
# FLAGS.growth_rate = 8
# FLAGS.depth = 40
FLAGS.total_blocks = 4
# FLAGS.keep_prob = 0.3
FLAGS.reduction = 0.5
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = True
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
	# Nesterov
#	[7, 3, 0.1, 10, 0.8, 16, 100, 32, 100, 100, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, False, 0.9],
#	[7, 3, 0.1, 10, 0.8, 16, 200, 32, 200, 200, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, False, 0.9],
#	[7, 3, 0.1, 10, 0.8, 16, 100, 32, 50, 100, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, False, 0.9],
#	[7, 3, 0.1, 10, 0.8, 16, 200, 32, 50, 200, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, False, 0.9],
#	[7, 3, 0.1, 10, 0.8, 16, 100, 32, 50, 100, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, True, 0.9],
#	[7, 3, 0.1, 10, 0.8, 16, 200, 32, 50, 200, '2.0,0.5', 3, 3, "single", 0.7, 5e-5, True, 0.9],

#	[7, 3, 0.001, 10, 0.8, 16, 100, 32, 100, 100, '2.0,0.5', 3, 3, "single", 0.7, 5e-4, False, 0.9],
#	[7, 3, 0.001, 10, 0.8, 16, 200, 32, 200, 200, '2.0,0.5', 3, 3, "single", 0.7, 5e-4, False, 0.9],
#	[7, 3, 0.01, 10, 0.8, 16, 100, 32, 100, 100, '2.0,0.5', 3, 3, "single", 0.7, 5e-4, False, 0.9],
#	[7, 3, 0.01, 10, 0.8, 16, 200, 32, 200, 200, '2.0,0.5', 3, 3, "single", 0.7, 5e-4, False, 0.9]

	# Adam
#	[7, 3, 0.0002, 24, 0.8, 37, 150, 32, 150, 150, '2.0,0.5', 3, 3, "single", 0.7, 0.0, False, 0.95, 0.999, 1e-3, 0.7],
#	[7, 3, 0.0002, 24, 0.8, 37, 150, 32, 150, 150, '2.0,0.5', 3, 3, "single", 0.7, 0.0, False, 0.95, 0.999, 1e-2, 0.7],
#	[7, 3, 0.0002, 24, 0.8, 37, 150, 32, 150, 150, '2.0,0.5', 3, 3, "single", 0.7, 0.0, False, 0.95, 0.999, 0.1, 0.7],
#	[7, 3, 0.0002, 24, 0.8, 37, 150, 32, 150, 150, '2.0,0.5', 3, 3, "single", 0.7, 0.0, False, 0.95, 0.999, 1.0, 0.7],
#	[7, 3, 0.0002, 24, 0.8, 37, 150, 32, 150, 150, '2.0,0.5', 3, 3, "single", 0.7, 1e-6, False, 0.95, 0.999, 1e-4, 0.7],

	[7, 3, 0.0003, 6, 0.8, 37, 200, 32, 200, 200, '2.0,1.5', '1.0,0.5', 3, 3, "avg", 0.7, False, 0.9, 0.999, 1e-3, 0.7, 1e-4, '30, 30'],  # -> run for longer
#	[7, 3, 0.0003, 6, 0.8, 37, 150, 32, 150, 150, '2.0,1.5', '1.0,0.5', 3, 3, "max", 0.7, False, 0.9, 0.999, 1e-3, 0.7, 1e-4, '30, 30'],
	[7, 3, 0.0003, 8, 0.8, 37, 150, 32, 150, 150, '2.0,1.5', '1.0,0.5', 3, 3, "avg", 0.7, False, 0.9, 0.999, 1e-3, 0.7, 1e-4, '45, 45'],
	[7, 3, 0.0003, 8, 0.8, 37, 150, 32, 150, 150, '2.0,1.5', '1.0,0.5', 3, 3, "max", 0.7, False, 0.9, 0.999, 1e-3, 0.7, 1e-4, '45, 45'],
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
	FLAGS.init_sigmas = config[10]
	FLAGS.comp_sigmas = config[11]
	FLAGS.init_order = config[12]
	FLAGS.comp_order = config[13]
	FLAGS.rfnn = config[14]
	FLAGS.bnorm_mom = config[15]
	FLAGS.bnorm_inc = config[16]
	FLAGS.beta1 = config[17]
	FLAGS.beta2 = config[18]
	FLAGS.epsilon = config[19]
	FLAGS.renorm = config[20]
	FLAGS.beta_wd = config[21]
	FLAGS.thetas = config[22]
	FLAGS.tol_var = 0.01
	FLAGS.t_max = 20
	FLAGS.t_min = 10
	FLAGS.X_dim = 433
#	FLAGS.X_dim = 336
#	FLAGS.Z_dim = 252

	FLAGS.xvalidation_folds = 4

	#    FLAGS.log_dir = '/home/nicolab/DATA/logs/NCCT_CTA/DenseNet/bests/'
#	FLAGS.log_dir = r'/home/hilbysfe/dev/DNN-for-CT-Classification/Training/logs/MIP2D/standard/' \
#	FLAGS.log_dir = r'D:\Adam Hilbert\CT_Classification\code\Training\logs\NCCT\standard\\' \
#	FLAGS.log_dir = r'D:\Adam Hilbert\CT_Classification\code\Training\logs\MIP2D\rfnn\single_scale\Adam\lsuv\\' \
	FLAGS.log_dir = r'E:\logs\MIP2D_affected\rfnn\SO\\' \
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
					+ str(FLAGS.bnorm_mom) \
					+ str(FLAGS.renorm) + '_' \
					+ str(FLAGS.beta1) + '-' + str(FLAGS.beta2) + '-' + str(FLAGS.epsilon) + '_red-' \
					+ str(FLAGS.reduction) + 'adaptWD-nobeta_' \
					+ str(FLAGS.thetas)
#					+ str(FLAGS.beta_wd) + '_' + '-wd' + str(FLAGS.beta_wd) + '_' \

#					+ str(FLAGS.nesterov_momentum) \
#					+ str(FLAGS.init_order) + '_' \
#					+ str(FLAGS.comp_order) + '_' \
	if FLAGS.bnorm_inc:
		FLAGS.log_dir += '_bnorm_inc' + '-0.75'
	#    FLAGS.checkpoint_dir = '/home/nicolab/DATA/checkpoints/NCCT_CTA/DenseNet/' \
#	FLAGS.checkpoint_dir = 'D:\Adam Hilbert\CT_Classification\code\Training\checkpoints\\' \
#						   + str(FLAGS.init_kernel) + 'x' + str(FLAGS.comp_kernel) + '_' \
#						   + str(FLAGS.learning_rate) + '_' \
#						   + str(FLAGS.growth_rate) + '_' \
#						   + str(FLAGS.depth) + '_' \
#						   + str(FLAGS.total_blocks) + '_' \
#						   + str(FLAGS.keep_prob) + '_' \
#						   + str(FLAGS.reduction)

	initialize_folders()
	train_ctnet(FLAGS, NUM_GPUS)
#	train_cifar(FLAGS, NUM_GPUS)
