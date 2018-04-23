from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shutil
from train_model_combination import train_ctnet

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

# Adam
FLAGS.beta1 = 0.9
FLAGS.beta2 = 0.999
FLAGS.epsilon = 1e-3

# LSUV
FLAGS.t_max = 20
FLAGS.t_min = 10
FLAGS.tol_var = 0.01

# General CNN
FLAGS.bases3d = False
FLAGS.init_kernel = 11
FLAGS.comp_kernel = 5
FLAGS.init_order = 3
FLAGS.comp_order = 3
FLAGS.bnorm_mom = 0.7
FLAGS.renorm = 0.7
FLAGS.beta_wd = 1e-4

FLAGS.keep_prob = 0.9

# DenseNet
FLAGS.total_blocks = 4
FLAGS.reduction = 0.5
FLAGS.model_type = 'DenseNet'
FLAGS.bc_mode = True

# Training
FLAGS.max_epochs = 40
FLAGS.batch_size = 32
FLAGS.xvalidation_folds = 4
FLAGS.trials = 1
FLAGS.weight_decay = 1e-4
FLAGS.refine = 30

# Data
FLAGS.normalization = True
FLAGS.X_dim = 433
FLAGS.bases3d = False
FLAGS.feats_use = 'Baseline_contscore'

FLAGS.input_num_units = 51
# Combination model
FLAGS.embedding = 128
FLAGS.num_of_layers = 0
FLAGS.num_of_neurons = []


# Logging
FLAGS.print_freq = 3
FLAGS.eval_freq = 1

configs = [

#	[0.0001, 8, 21, 200, 200, '2.0,1.5', '1.0,0.5', "learn_sq", False, '45, 45', 'mrs',
#	 '11x5_0.0003_8_21_4_0.9_32_learn_sq_2.0,1.51.0,0.5_200_200_100_0.70.7_0.9-0.999-0.001_red-0.5adaptWD-nobeta_45, 45',
#	 0.8, 1e-4, 128, [512, 64]],
#	[0.0001, 8, 21, 200, 200, '2.0,1.5', '1.0,0.5', "cnn", False, '45, 45', 'mrs',
#	 '11x5_0.0003_8_21_4_0.9_32_cnn_2.0,1.51.0,0.5_200_200_100_0.70.7_0.9-0.999-0.001_red-0.5adaptWD-nobeta_45, 45',
#	 0.8, 1e-4, 128, [512, 64]],


	[0.0001, 8, 21, 200, 200, '2.0,1.5', '1.0,0.5', "learn_sq", False, '45, 45', 'tici_imp',
	 '11x5_0.0003_8_21_4_0.9_32_learn_sq_2.0,1.51.0,0.5_200_200_100_0.70.7_0.9-0.999-0.001_red-0.5adaptWD-nobeta_45, 45',
	 0.8, 1e-4, 128, [512, 64]],
	[0.0001, 8, 21, 200, 200, '2.0,1.5', '1.0,0.5', "cnn", False, '45, 45', 'tici_imp',
	 '11x5_0.0003_8_21_4_0.9_32_cnn_2.0,1.51.0,0.5_200_200_100_0.70.7_0.9-0.999-0.001_red-0.5adaptWD-nobeta_45, 45',
	 0.8, 1e-4, 128, [512, 64]]

]

for i, config in enumerate(configs):

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
	FLAGS.label = config[10]
	FLAGS.keep_prob_comb = config[12]
	FLAGS.weight_decay = config[13]
	FLAGS.embedding = config[14]
	FLAGS.num_of_neurons = config[15]
	FLAGS.num_of_layers = len(config[15])

	FLAGS.datapath = r'D:\Adam Hilbert\Data\data_binaries\MIP2D\\' + FLAGS.label

	FLAGS.log_dir = r'D:\Experiments\logs\MIP2D_' + FLAGS.label + '\\comb\\' + config[7] + '\\' + \
		str(FLAGS.learning_rate) + '_' + str(FLAGS.num_of_neurons) + '_' + str(FLAGS.embedding) \
		+ '_' + str(FLAGS.max_epochs) \
		+ '_' + str(FLAGS.keep_prob_comb) \
		+ '_' + str(FLAGS.weight_decay) + '_bn0.9_bias_refine'
	FLAGS.checkpoint_dir = r'D:\Experiments\checkpoints\MIP2D_' + FLAGS.label + '\\comb\\' + config[7] + '\\' + \
		str(FLAGS.learning_rate) + '_' + str(FLAGS.num_of_neurons) + '_' + str(FLAGS.embedding) \
		+ '_' + str(FLAGS.max_epochs) \
		+ '_' + str(FLAGS.keep_prob_comb) \
		+ '_' + str(FLAGS.weight_decay) + '_bn0.9_bias_refine'
	FLAGS.stat_dir = r'D:\Experiments\stats\MIP2D_' + FLAGS.label + '\\comb\\' + config[7] + '\\' + \
		str(FLAGS.learning_rate) + '_' + str(FLAGS.num_of_neurons) + '_' + str(FLAGS.embedding) \
		+ '_' + str(FLAGS.max_epochs) \
		+ '_' + str(FLAGS.keep_prob_comb) \
		+ '_' + str(FLAGS.weight_decay) + '_bn0.9_bias_refine'

	FLAGS.log_dir_vars = r'D:\Experiments\logs\MIP2D_' + FLAGS.label + '\\NN'
	FLAGS.stat_dir_vars = r'D:\Experiments\stats\MIP2D_' + FLAGS.label + '\\NN'
	FLAGS.checkpoint_dir_vars = r'D:\Experiments\checkpoints\MIP2D_' + FLAGS.label + '\\NN'

	FLAGS.checkpoint_dir_cnn = \
		r'D:\Experiments\checkpoints\MIP2D_' + config[10] + '\\' + 'rfnn' + '\\' + 'Shallow' + '\\' + config[11]
	FLAGS.stat_dir_cnn = \
		r'D:\Experiments\stats\MIP2D_' + config[10] + '\\' + 'rfnn' + '\\' + 'Shallow' + '\\' + config[11]


	initialize_folders()
	train_ctnet(FLAGS)
