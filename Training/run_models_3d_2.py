from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
from train_model import train_ctnet

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
FLAGS.max_epochs = 40
FLAGS.xvalidation_folds = 4
FLAGS.trials = 1
# Adam
FLAGS.beta1 = 0.9
FLAGS.beta2 = 0.999
FLAGS.epsilon = 1e-3

# LSUV
FLAGS.t_max = 10
FLAGS.t_min = 5
FLAGS.tol_var = 0.01

# Data
#FLAGS.cifar_path = '/home/hilbysfe/dev/DNN-for-CT-Classification/Training/cifar-10-batches-py'
#FLAGS.cifar_path = r'D:\Adam Hilbert\CT_Classification\code\Training\cifar10\cifar-10-batches-py'
FLAGS.normalization = True
# FLAGS.X_dim = 433
FLAGS.X_dim = 336
FLAGS.Z_dim = 252

# General CNN
FLAGS.bases3d = True
FLAGS.init_kernel = 11
FLAGS.comp_kernel = 5
FLAGS.keep_prob = 0.9
FLAGS.init_order = 3
FLAGS.comp_order = 3
FLAGS.bnorm_mom = 0.7
FLAGS.renorm = 0.7
FLAGS.beta_wd = 1e-4

# DenseNet
FLAGS.total_blocks = 2
FLAGS.reduction = 1.0
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = False

# Logging
FLAGS.print_freq = 3
FLAGS.eval_freq = 1

NUM_GPUS = 1

configs = [
#	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "learn_sq", False, '60, 60', '60, 60', 'tici_imp', 16, False, '0'],
#	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "learn_sq", False, '60, 60', '60, 60', 'mrs', 16, False, '0']

	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "cnn", False, '60, 60', '60, 60', 'mrs', 16, True, '0'],
	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "cnn", False, '60, 60', '60, 60', 'tici_imp', 16, True, '0']

#	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "single", False, '60, 60', '60, 60', 'tici_imp', 16, False, '0'],
#	[0.0003, 2, 5, 200, 200, '2.0,1.5', '1.0,0.5', "single", False, '60, 60', '60, 60', 'mrs', 16, False, '0']
]


for config in configs:
	FLAGS.datapath = r'/home/hilbysfe/DATA/data_binaries/NCCT/'

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
	FLAGS.phis = config[10]
	FLAGS.datapath = FLAGS.datapath + config[11]
	FLAGS.batch_size = config[12]
	FLAGS.normalization = config[13]
	FLAGS.gpu_list = config[14]

	FLAGS.mean_dir = r'/home/hilbysfe/DATA/Experiments/means/' + config[11] + '/'

	FLAGS.log_dir = r'/home/hilbysfe/DATA/Experiments/logs/NCCT_' + config[11] \
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

	FLAGS.checkpoint_dir = r'/home/hilbysfe/DATA/Experiments/checkpoints/NCCT_' + config[11] \
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

	FLAGS.stat_dir = r'/home/hilbysfe/DATA/Experiments/stats/NCCT_' + config[11] \
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
	train_ctnet(FLAGS, NUM_GPUS)
