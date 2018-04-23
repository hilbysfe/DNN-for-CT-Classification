# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:00:03 2017

@author: laramos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

import Data_Preprocessing as pp
import tensorflow as tf
from Utils import utils_final
import random
from sklearn.metrics import roc_auc_score
from scipy import interp
from sklearn.metrics import roc_curve
import re
import os
from sklearn.metrics import auc

"""
-----------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------
"""


def Binarize_Labels(Y):
	Y2 = np.zeros((Y.shape[0], 2))
	for i in range(0, Y.shape[0]):
		if Y[i] == 1:
			Y2[i, 0] = 0
			Y2[i, 1] = 1
		else:
			Y2[i, 0] = 1
			Y2[i, 1] = 0
	return (Y2)

def Search_Pats(sub_id, points, labels):
	for cont, key in enumerate(points):
		pat_num = key
		sub_id.append(re.findall('(R[0-9]+)', pat_num))
		labels[cont] = points[key]

	for i in range(len(sub_id)):
		if sub_id[i] not in sub_id_data:
			print("Train Patient not found", sub_id[i])

	sub_id = np.array(sub_id)
	index = np.where(sub_id_data == sub_id)
	index = index[1]
	labels = Binarize_Labels(labels)
	return (index, labels)

def Connect_Data(points_train, points_test, points_val, sub_id_data):
	sub_id_train = list()
	sub_id_test = list()
	sub_id_val = list()
	labels_train = np.zeros(len(points_train))
	labels_test = np.zeros(len(points_test))
	labels_val = np.zeros(len(points_val))

	index_train, labels_train = Search_Pats(sub_id_train, points_train, labels_train)
	index_test, labels_test = Search_Pats(sub_id_test, points_test, labels_test)
	index_val, labels_val = Search_Pats(sub_id_val, points_val, labels_val)

	return (index_train, index_test, index_val, labels_train, labels_test, labels_val)


class FLAGS():
	def __init__(self):
		return

training_points, test_points, validation_points = utils_final.read_dataset(
	r"D:\Adam Hilbert\Data\data_binaries\MIP2D\mrs")

sub_id_data = np.load(r'D:\Adam Hilbert\Data\Registry\sub_id_complete.npy')

path_data = r'D:\Adam Hilbert\Data\Registry\Imputed_Adam.csv'
Path_Results = r"D:\Experiments\stats\Clinical"

feats_use = 'Baseline_contscore'  # Done
# feats_use='Baseline_binscore' #Done
# feats_use='All_vars_binscore' #Done
# feats_use='All_vars_contscore' #Done
# feats_use='Knowledge_baseline'
# feats_use='Knowledge_all'

label_use = 'mrs'

FLAGS = FLAGS()
FLAGS.max_epochs = 300
FLAGS.batch_size = 32
FLAGS.xvalidation_folds = 4
FLAGS.trials = 3
FLAGS.checkpoint_dir = r'D:\Experiments\checkpoints\MIP2D_' + label_use + '\\NN'
FLAGS.stat_dir = r'D:\Experiments\stats\MIP2D_' + label_use + '\\NN'


[X, cols, center, vals_mask] = pp.Fix_Dataset_csv(path_data, feats_use)

# hidden_num_units_1 = 194
hidden_num_units_1 = 78
hidden_num_units_2 = 156
hidden_num_units_3 = 78

learning_rate = 0.001
mean_tprr = 0.0
mean_tprn = 0.0
mean_fpr = np.linspace(0, 1, 100)

tf.set_random_seed(256)

# define placeholders
input_num_units = X.shape[1]
x = tf.placeholder(tf.float32, [FLAGS.batch_size, input_num_units], name='input-layer')
y = tf.placeholder(tf.float32, [FLAGS.batch_size, 2])
prob = tf.placeholder_with_default(1.0, shape=())

with tf.variable_scope('NN'):

	weights = {
		'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units_1]), name='wh_1'),
		'hidden2': tf.Variable(tf.random_normal([hidden_num_units_1, hidden_num_units_2])),
		'hidden3': tf.Variable(tf.random_normal([hidden_num_units_2, hidden_num_units_3])),
		'output': tf.Variable(tf.random_normal([hidden_num_units_1, 2]))
	}
	biases = {
		'hidden': tf.Variable(tf.random_normal([hidden_num_units_1])),
		'hidden2': tf.Variable(tf.random_normal([hidden_num_units_2])),
		'hidden3': tf.Variable(tf.random_normal([hidden_num_units_3])),
		'output': tf.Variable(tf.random_normal([2]))
	}

	# layer 1
	hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'], name='h1_add')
	hidden_layer = tf.nn.sigmoid(hidden_layer, name='h1_sg')
	hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h1_dp')
	# layer 2
	hidden_layer = tf.add(tf.matmul(hidden_layer, weights['hidden2']), biases['hidden2'], name='h2_add')
	hidden_layer = tf.nn.sigmoid(hidden_layer, name='h2_sg')
	hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h2_dp')
	# layer 3
	hidden_layer = tf.add(tf.matmul(hidden_layer, weights['hidden3']), biases['hidden3'], name='h3_add')
	hidden_layer = tf.nn.sigmoid(hidden_layer, name='h3_sg')
	hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h3_dp')
	# output
	output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
	probs = tf.nn.softmax(output_layer, name="softmax_tensor")

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

vars_training = []
vars_testing = []
vars_validation = []
for i in range(FLAGS.xvalidation_folds):
	index_train, index_test, index_val, Y_train, Y_test, Y_val = Connect_Data(training_points[i], test_points[i],
																			  validation_points[i], sub_id_data)

	scaler = preprocessing.StandardScaler().fit(X[index_train, :])
	X_train = scaler.transform(X[index_train, :])
	X_test = scaler.transform(X[index_test, :])
	X_val = scaler.transform(X[index_val, :])

	vars_training.append(X_train)
	vars_testing.append(X_test)
	vars_validation.append(X_val)

dataset = utils_final.DataSetCombined(training_points, test_points, validation_points,
							  vars_training, vars_testing, vars_validation)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	# ========= CROSS-VALIDATE RESULTS ==========
	best_trials = []
	for f in range(FLAGS.xvalidation_folds):
		print('X-val fold: ' + str(f))
		training_steps = int(np.ceil(2 * dataset.Training.num_examples / FLAGS.batch_size))
		validation_steps = int(np.ceil(2 * dataset.Validation.num_examples / FLAGS.batch_size))
		test_steps = int(np.ceil(2 * dataset.Test.num_examples / FLAGS.batch_size))

		# ======== X TRIALS =========
		best_trial = 0
		best_loss = 1000.0
		for trial in range(FLAGS.trials):
			print('Trial: ' + str(trial))

			# Check if trial already run
			if tf.gfile.Exists(FLAGS.stat_dir + '\\min_loss_' + str(f) + '_' + str(trial) + '.npy'):
				min_loss = np.load(FLAGS.stat_dir + '\\min_loss_' + str(f) + '_' + str(trial) + '.npy')

				if min_loss < best_loss:
					best_trial = trial
					best_loss = min_loss

				continue

			sess.run(tf.global_variables_initializer())

			max_acc = 0
			min_loss = 1000.0
			# ==== TRAIN MODEL TILL max_epochs ====
			print('Training model...')
			for i in range(FLAGS.max_epochs):
				avg_loss = 0
				# ------------ TRAIN -------------
				for step in range(training_steps):
					x_batch, y_batch = dataset.Training.next_batch(FLAGS.batch_size, only_vars=True)
					_, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch, prob: 0.7})

					avg_loss += (c / training_steps)

				# ------------ VALIDATION & PRINT -------------
				avg_loss_entropy = 0.0
				for step in range(validation_steps):
					x_batch, y_batch = dataset.Validation.next_batch(FLAGS.batch_size, only_vars=True)
					loss_entropy = cost.eval(
							feed_dict={x: x_batch, y: y_batch, prob: 1.0})
					avg_loss_entropy += loss_entropy / validation_steps
				print("Validation CE: %.4f, Training CE: %.4f" % (avg_loss_entropy, avg_loss))

				# Save bets model so far
				if avg_loss_entropy < min_loss and i > FLAGS.max_epochs * 0.5:
					min_loss = avg_loss_entropy

					if not tf.gfile.Exists(
							os.path.join(FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(trial))):
						tf.gfile.MakeDirs(
							os.path.join(FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(trial)))
					checkpoint_path = FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(
						trial) + '\\' + 'best_model'
					saver.save(sess, checkpoint_path)

			if not tf.gfile.Exists(FLAGS.stat_dir):
				tf.gfile.MakeDirs(FLAGS.stat_dir)
			np.save(FLAGS.stat_dir + '\\min_loss_' + str(f) + '_' + str(trial) + '.npy', min_loss)
			if min_loss < best_loss:
				best_trial = trial
				best_loss = min_loss
				best_acc = max_acc

			dataset.Training.reset()
			dataset.Validation.reset()

		best_trials.append(best_trial)
		dataset.next_fold()

	# ===== LOAD BEST MODELS FOR ALL FOLDS AND COMPUTE STATISTICS FOR TEST SET =====
	dataset.reset()
	cv_acc_list = []
	cv_auc_list = []
	mean_fpr = np.linspace(0, 1, 100)
	cv_tprs_list = []
	for f in range(FLAGS.xvalidation_folds):
		model_path = FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(best_trials[f])
		new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'best_model.meta'))
		new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

		# Save parameters
		weights_dict = sess.run(weights)
		bias_dict = sess.run(biases)

		with open(os.path.join(FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(best_trials[f]), 'weights.npy'),
						   'wb') as output:
			pickle.dump(weights_dict, output)

		with open(os.path.join(FLAGS.checkpoint_dir + '\\' + str(f) + '\\' + str(best_trials[f]), 'bias.npy'),
						   'wb') as output:
			pickle.dump(bias_dict, output)

		# ======== ROC Analysis ==========
		iters = 10
		aucs = []
		accs = []
		tprs = []
		for k in range(iters):
			avg_loss_entropy = 0.0
			test_acc = 0.0
			softmax_whole = []
			labels_whole = []
			for step in range(test_steps):
				x_batch, y_batch = dataset.Test.next_batch(FLAGS.batch_size, only_vars=True)
				_, softmax, acc_s = sess.run([cost, probs, accuracy],
									feed_dict={x: x_batch, y: y_batch, prob: 1.0})
				softmax_whole.append(softmax)
				labels_whole.append(y_batch)
				test_acc += (acc_s / test_steps)
			softmax_whole = np.reshape(softmax_whole, (np.shape(softmax_whole)[0] * np.shape(softmax_whole)[1], 2))
			labels_whole = np.reshape(labels_whole, (np.shape(labels_whole)[0] * np.shape(labels_whole)[1], 2))

			# Compute ROC curve and ROC area for each class
			fpr, tpr, _ = roc_curve(labels_whole[:,1], softmax_whole[:,1])
			tprs.append(interp(mean_fpr, fpr, tpr))
			tprs[-1][0] = 0.0

			auc_k = auc(fpr, tpr)
			aucs.append(auc_k)
			accs.append(test_acc)

		cv_auc_list.append(np.mean(aucs))
		cv_acc_list.append(np.mean(accs))
		cv_tprs_list.append(np.mean(tprs, axis=0))

		dataset.next_fold()

	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

	avg_auc = np.mean(np.array(cv_auc_list))
	std_auc = np.std(np.array(cv_auc_list))

	avg_acc = np.mean(np.array(cv_acc_list))
	std_acc = np.std(np.array(cv_acc_list))

	avg_tpr = np.mean(np.array(cv_tprs_list), axis=0)
	avg_tpr[-1] = 1.0
	std_tpr = np.std(np.array(cv_tprs_list), axis=0)

	plt.plot(mean_fpr, avg_tpr, color='b',
			 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (avg_auc, std_auc),
			 lw=2, alpha=.8)

	tprs_upper = np.minimum(avg_tpr + std_tpr, 1)
	tprs_lower = np.maximum(avg_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

	# Save statistics
	if not tf.gfile.Exists(FLAGS.stat_dir):
		tf.gfile.MakeDirs(FLAGS.stat_dir)
	np.save(os.path.join(FLAGS.stat_dir + '\\', 'cv_tprs_list.npy'), np.array(cv_tprs_list))
	np.save(os.path.join(FLAGS.stat_dir + '\\', 'mean_fpr.npy'), mean_fpr)
	np.savetxt(os.path.join(FLAGS.stat_dir + '\\', 'cv_auc_list.csv'), np.array(cv_auc_list), delimiter=",",
			   fmt='%.5ef')
	np.savetxt(os.path.join(FLAGS.stat_dir + '\\', 'cv_acc_list.csv'), np.array(cv_acc_list), delimiter=",",
			   fmt='%.5ef')

	print('Acc/std/AUC/std : %s/%s/%s/%s' % (avg_acc, std_acc, avg_auc, std_auc))

