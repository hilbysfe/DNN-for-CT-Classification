from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
from train_cifar import train_cifar
from train_model_cnn import train_ctnet_cnn

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
FLAGS.max_epochs = 100
FLAGS.batch_size = 32
FLAGS.xvalidation_folds = 4
FLAGS.trials = 3
# Adam
FLAGS.beta1 = 0.9
FLAGS.beta2 = 0.999
FLAGS.epsilon = 1e-3

# LSUV
FLAGS.t_max = 20
FLAGS.t_min = 10
FLAGS.tol_var = 0.01

# Data
#FLAGS.cifar_path = '/home/hilbysfe/dev/DNN-for-CT-Classification/Training/cifar-10-batches-py'
#FLAGS.cifar_path = r'D:\Adam Hilbert\CT_Classification\code\Training\cifar10\cifar-10-batches-py'
FLAGS.normalization = True
FLAGS.X_dim = 433
#	FLAGS.X_dim = 336
#	FLAGS.Z_dim = 252

# General CNN
FLAGS.bases3d = False
FLAGS.init_kernel = 11
FLAGS.comp_kernel = 5
FLAGS.keep_prob = 0.8
FLAGS.init_order = 3
FLAGS.comp_order = 3
FLAGS.bnorm_mom = 0.7
FLAGS.renorm = 0.7
FLAGS.beta_wd = 1e-4

# DenseNet
FLAGS.total_blocks = 4
FLAGS.reduction = 0.5
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = True

# Logging
FLAGS.print_freq = 3
FLAGS.eval_freq = 1

configs = [
#	SHAllOW
#	[0.0003, 8, 21, 200, 200, '2.0', '0.5', "cnn", False, '45, 45', 'affected_side'],
	[0.0003, 8, 21, 200, 200, '2.0', '0.5', "cnn", False, '45, 45', 'collaterals_imp'],
	[0.0003, 8, 21, 200, 200, '2.0', '0.5', "cnn", False, '45, 45', 'nihss_imp'],
	[0.0003, 8, 21, 200, 200, '2.0', '0.5', "cnn", False, '45, 45', 'tici_imp'],
	[0.0003, 8, 21, 200, 200, '2.0', '0.5', "cnn", False, '45, 45', 'mrs'],


#	DEEP
#	[0.0003, 8, 37, 200, 200, '2.0,1.5', '1.0,0.5', "single", False, '45, 45'],
#	[0.0003, 8, 37, 200, 200, '2.0,1.5', '1.0,0.5', "learn_sq", False, '45, 45'],
#	[0.0003, 8, 37, 200, 200, '2.0,1.5', '1.0,0.5', "avg", False, '45, 45'],
#	[0.0003, 8, 37, 200, 200, '2.0,1.5', '1.0,0.5', "max", False, '45, 45']
]


for config in configs:
	FLAGS.datapath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D\\'

	FLAGS.learning_rate = config[0]
	FLAGS.growth_rate = config[1]
	FLAGS.depth = config[2]
	FLAGS.reduce_lr_epoch_1 = config[3]
	FLAGS.reduce_lr_epoch_2 = config[4]
	FLAGS.init_sigmas = config[5]
	FLAGS.comp_sigmas = config[6]
	FLAGS.rfnn = config[7]
	FLAGS.bnorm_inc = config[8]
	FLAGS.thetas = config[9]
	FLAGS.datapath = FLAGS.datapath + config[10]

	FLAGS.log_dir = r'D:\Experiments\logs\MIP2D_' + config[10] + '\\cnn\Shallow\\' \
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
	if FLAGS.bnorm_inc:
		FLAGS.log_dir += '_bnorm_inc' + '-0.75'

	FLAGS.checkpoint_dir = r'D:\Experiments\checkpoints\MIP2D_' + config[10] + '\\cnn\Shallow\\' \
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
	if FLAGS.bnorm_inc:
		FLAGS.checkpoint_dir += '_bnorm_inc' + '-0.75'

	FLAGS.stat_dir = r'D:\Experiments\stats\MIP2D_' + config[10] + '\\cnn\Shallow\\' \
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
	if FLAGS.bnorm_inc:
		FLAGS.stat_dir += '_bnorm_inc' + '-0.75'


	initialize_folders()
	train_ctnet_cnn(FLAGS, NUM_GPUS)
#	train_cifar(FLAGS, NUM_GPUS)
