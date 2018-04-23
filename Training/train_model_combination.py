
from Utils import utils
from Utils import utils_final
from Models.combination import CombinationModel
from Models.densenet import DenseNet
from Models.densenet3d import DenseNet3d
from Models.RFNN_densenet import RFNNDenseNet
from Models.RFNN_densenet3d import RFNNDenseNet3D

from Utils.training_utils import average_gradients
from Utils.training_utils import tower_loss_dense
from Utils.training_utils import tower_loss_dense_wd
from Utils.training_utils import tower_loss
from Utils.training_utils import tower_accuracy
from Utils.training_utils import show_kernels

import numpy as np
from scipy import interp
import tensorflow as tf
import pickle
import os
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import Data_Preprocessing as pp
from sklearn import preprocessing
import re



import gradient_checkpointing_master.memory_saving_gradients as memory_saving_gradients
from tensorflow.python.ops import gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', gate_gradients=True, **kwargs)
gradients.__dict__["gradients"] = gradients_memory


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

def Search_Pats(sub_id, points, labels, sub_id_data):
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

	index_train, labels_train = Search_Pats(sub_id_train, points_train, labels_train, sub_id_data)
	index_test, labels_test = Search_Pats(sub_id_test, points_test, labels_test, sub_id_data)
	index_val, labels_val = Search_Pats(sub_id_val, points_val, labels_val, sub_id_data)

	return (index_train, index_test, index_val, labels_train, labels_test, labels_val)

def train_ctnet(FLAGS):
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			# ====== DEFINE SPACEHOLDERS ======
			with tf.name_scope('input'):
				if FLAGS.bases3d:
					image_batch = tf.placeholder(tf.float32,
												 [FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1],
												 name='x-input')
				else:
					image_batch = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 1],
												 name='x-input')
			with tf.name_scope('alg-parameters'):
				is_training_cnn = tf.constant(False, tf.bool, name='is-training')
				is_training_comb = tf.placeholder(tf.bool, name='is-training-combination')
				is_refine = tf.placeholder(tf.bool, name='is-refine')

# ===========================================================================
# ========================= CNN MODEL =======================================
# ===========================================================================
			print('Defining cnn model...')
			init_sigmas = [float(x) for x in FLAGS.init_sigmas.split(',')]
			comp_sigmas = [float(x) for x in FLAGS.comp_sigmas.split(',')]
			thetas = [float(x) for x in FLAGS.thetas.split(',')]

			if FLAGS.rfnn == "cnn":
				cnn_model = DenseNet(
					growth_rate=FLAGS.growth_rate,
					depth=FLAGS.depth,
					total_blocks=FLAGS.total_blocks,
					keep_prob=FLAGS.keep_prob,
					model_type=FLAGS.model_type,
					is_training=is_training_cnn,
					init_kernel=FLAGS.init_kernel,
					comp_kernel=FLAGS.comp_kernel,
					bnorm_momentum=FLAGS.bnorm_mom,
					renorm=FLAGS.renorm,
					beta_wd=FLAGS.beta_wd,
					reduction=FLAGS.reduction,
					bc_mode=FLAGS.bc_mode,
					n_classes=2)
			else:
				cnn_model = RFNNDenseNet(
					growth_rate=FLAGS.growth_rate,
					depth=FLAGS.depth,
					total_blocks=FLAGS.total_blocks,
					keep_prob=FLAGS.keep_prob,
					model_type=FLAGS.model_type,
					is_training=is_training_cnn,
					init_kernel=FLAGS.init_kernel,
					comp_kernel=FLAGS.comp_kernel,
					init_sigmas=init_sigmas,
					comp_sigmas=comp_sigmas,
					init_order=FLAGS.init_order,
					comp_order=FLAGS.comp_order,
					thetas=thetas,
					rfnn=FLAGS.rfnn,
					bnorm_momentum=FLAGS.bnorm_mom,
					renorm=FLAGS.renorm,
					beta_wd=FLAGS.beta_wd,
					reduction=FLAGS.reduction,
					bc_mode=FLAGS.bc_mode,
					n_classes=2)
			print('Defining model...done.')

		with tf.device('/gpu:0'):
			# ====== INFERENCE ======
			with tf.name_scope('%s_%d' % ('tower', 0)):
				# Calculate predictions
				logits, penultimate = cnn_model.inference(image_batch)

		# ========= LOAD BEST MODELS PER FOLD ==========
		cnn_best_trials = []
		for f in range(FLAGS.xvalidation_folds):
			# ======== X TRIALS =========
			best_trial = 0
			best_loss = 1000.0
			best_acc = 0.0
			for trial in range(3):
				# Check if trial already run
				if tf.gfile.Exists(FLAGS.stat_dir_cnn + '/min_loss_' + str(f) + '_' + str(trial) + '.npy'):
					min_loss = np.load(FLAGS.stat_dir_cnn + '/min_loss_' + str(f) + '_' + str(trial) + '.npy')
					max_acc = np.load(FLAGS.stat_dir_cnn + '/max_acc_' + str(f) + '_' + str(trial) + '.npy')

					if min_loss < best_loss:
						best_trial = trial
						best_loss = min_loss
						best_acc = max_acc
					elif min_loss == best_loss and max_acc > best_acc:
						best_trial = trial
						best_acc = max_acc

					continue
				else:
					print('Failed to load: ' + FLAGS.stat_dir_cnn + '/min_loss_' + str(f) + '_' + str(trial) + '.npy')
					break

			cnn_best_trials.append(best_trial)

# ===========================================================================
# ====================== Neural Network MODEL ===============================
# ===========================================================================
		with tf.device('/cpu:0'):
			hidden_num_units_1 = 78
			hidden_num_units_2 = 156
			hidden_num_units_3 = 78
			input_num_units = 51

			# define placeholders
			x_vars = tf.placeholder(tf.float32, [FLAGS.batch_size, input_num_units], name='input-layer')

			with tf.variable_scope('NN'):
				nn_weights = {
					'hidden': tf.get_variable(shape=[input_num_units, hidden_num_units_1], initializer=tf.random_normal_initializer, name='wh_1'),
					'hidden2': tf.get_variable(shape=[hidden_num_units_1, hidden_num_units_2], initializer=tf.random_normal_initializer, name='wh_2'),
					'hidden3': tf.get_variable(shape=[hidden_num_units_2, hidden_num_units_3], initializer=tf.random_normal_initializer, name='wh_3'),
					'output': tf.get_variable(shape=[hidden_num_units_1, 2], initializer=tf.random_normal_initializer, name='wh_4')
				}
				nn_biases = {
					'hidden': tf.get_variable(shape=[hidden_num_units_1], initializer=tf.constant_initializer(0.0), name='b_1'),
					'hidden2': tf.get_variable(shape=[hidden_num_units_2], initializer=tf.constant_initializer(0.0), name='b_2'),
					'hidden3': tf.get_variable(shape=[hidden_num_units_3], initializer=tf.constant_initializer(0.0), name='b_3'),
					'output': tf.get_variable(shape=[2], initializer=tf.constant_initializer(0.0), name='b_4')
				}
		with tf.device('/gpu:0'):
			with tf.variable_scope('NN'):
				# ====== INFERENCE ======
				# layer 1
				hidden_layer = tf.add(tf.matmul(x_vars, nn_weights['hidden']), nn_biases['hidden'], name='h1_add')
				hidden_layer = tf.nn.sigmoid(hidden_layer, name='h1_sg')
				hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=1.0, name='h1_dp')
				# layer 2
				hidden_layer = tf.add(tf.matmul(hidden_layer, nn_weights['hidden2']), nn_biases['hidden2'], name='h2_add')
				hidden_layer = tf.nn.sigmoid(hidden_layer, name='h2_sg')
				hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=1.0, name='h2_dp')
				# layer 3
				hidden_layer = tf.add(tf.matmul(hidden_layer, nn_weights['hidden3']), nn_biases['hidden3'], name='h3_add')
				nn_output = tf.cond(is_refine,
									lambda: tf.nn.sigmoid(hidden_layer, name='h3_sg'),
									lambda: tf.stop_gradient(tf.nn.sigmoid(hidden_layer, name='h3_sg')))

		# ========= LOAD BEST MODELS PER FOLD ==========
		nn_best_trials = []
		for f in range(FLAGS.xvalidation_folds):
			# ======== X TRIALS =========
			best_trial = 0
			best_loss = 1000.0
			best_acc = 0.0
			for trial in range(3):
				# Check if trial already run
				if tf.gfile.Exists(FLAGS.stat_dir_vars + '/min_loss_' + str(f) + '_' + str(trial) + '.npy'):
					min_loss = np.load(FLAGS.stat_dir_vars + '/min_loss_' + str(f) + '_' + str(trial) + '.npy')

					if min_loss < best_loss:
						best_trial = trial
						best_loss = min_loss

					continue
				else:
					print('Failed to load: ' + FLAGS.stat_dir_vars + '/min_loss_' + str(f) + '_' + str(trial) + '.npy')
					break

			nn_best_trials.append(best_trial)

# ===========================================================================
# ====================== Combination MODEL ==================================
# ===========================================================================

		gl_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# ====== LOAD DATASET ======
		print('Loading Dataset...')
		training_points, test_points, validation_points = utils_final.read_dataset(FLAGS.datapath)

		[X, _, _, _] = pp.Fix_Dataset_csv(r'D:\Adam Hilbert\Data\Registry\Imputed_Adam.csv', FLAGS.feats_use)
		sub_id_data = np.load(r'D:\Adam Hilbert\Data\Registry\sub_id_complete.npy')

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
											  vars_training, vars_testing, vars_validation, normalize=True)

		print('Loading Dataset...done.')

		# ====== MODEL DEFINITION ======
		print('Defining model...')
		model = CombinationModel(
			embedding_neurons=FLAGS.embedding,
			num_of_layers=FLAGS.num_of_layers,
			num_of_neurons=FLAGS.num_of_neurons,
			keep_prob = FLAGS.keep_prob_comb,
			is_training=is_training_comb)
		print('Defining model...done.')

		# ====== DEFINE LOSS, ACCURACY TENSORS ======
		print('Defining necessary OPs...')
		opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

		# ====== FORWARD PASS ======
		with tf.variable_scope(tf.get_variable_scope()):#, reuse=True):
			with tf.device('/gpu:0'):
				# ====== INFERENCE ======
				with tf.name_scope('tower0') as scope:
					# Stop the gradient
					cnn_output_sg = tf.cond(is_refine,
										lambda: tf.nn.sigmoid(penultimate),
										lambda: tf.stop_gradient(tf.nn.sigmoid(penultimate)))

					# Define placeholder for labels
					label_batch = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='y-input')
					# Calculate predictions
					logits = model.inference([nn_output, cnn_output_sg])
					train_vars = [var for var in tf.trainable_variables() if 'Combination_model' in var.name]
					combined_loss, l2_loss, entropy = tower_loss_dense_wd(logits, label_batch, FLAGS.weight_decay, train_vars, scope)
					combined_accuracy, _, combined_scores = tower_accuracy(logits, label_batch)

					# Reuse variables for the next tower.
					tf.get_variable_scope().reuse_variables()

					# Retain the summaries from the final tower.
					summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

					# Calculate the gradients for the batch of data on this tower.
#					grads = tf.gradients(combined_loss, train_vars)
#					grads_and_vars = list(zip(grads, train_vars))

					grads = tf.gradients(combined_loss, tf.trainable_variables())
					grads_and_vars = list(zip(grads, tf.trainable_variables()))


		# ====== BACKWARD PASS ======
		with tf.device('/cpu:0'):
			print('Defining necessary OPs...done.')

			# ====== ADD SUMMARIES ======
			# Gradients
			for grad, var in grads_and_vars:
				if grad is not None:
					summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
			# Trainable variables
			for var in train_vars:
				summaries.append(tf.summary.histogram(var.op.name, var))

			# ====== UPDATE VARIABLES ======
			print('Defining update OPs...')
			# Apply the gradients to adjust the shared variables.
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = opt.apply_gradients(grads_and_vars, global_step=gl_step)
			print('Defining update OPs...done.')

			# ====== SAVING OPS ======
			# Create a saver.
			saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
			# Build the summary operation from the last tower summaries.
			summary_op = tf.summary.merge(summaries)

			# ====== DEFINE SESSION AND OPTIMIZE ======
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			# ========= CROSS-VALIDATE RESULTS ==========
			best_trials = []
			for f in range(FLAGS.xvalidation_folds):
				print('X-val fold: ' + str(f))
				test_steps = int(np.ceil(2 * dataset.Test.num_examples / FLAGS.batch_size))
				training_steps = int(np.ceil(2 * dataset.Training.num_examples / FLAGS.batch_size))
				validation_steps = int(np.ceil(2 * dataset.Validation.num_examples / FLAGS.batch_size))
				# ======== X TRIALS =========
				best_trial = 0
				best_loss = 1000.0
				best_acc = 0.0
				for trial in range(FLAGS.trials):
					print('Trial: ' + str(trial))

					# Check if trial already run
					if tf.gfile.Exists(FLAGS.stat_dir + '/min_loss_' + str(f) + '_' + str(trial) + '.npy'):
						min_loss = np.load(FLAGS.stat_dir + '/min_loss_' + str(f) + '_' + str(trial) + '.npy')
						max_acc = np.load(FLAGS.stat_dir + '/max_acc_' + str(f) + '_' + str(trial) + '.npy')

						if min_loss < best_loss:
							best_trial = trial
							best_loss = min_loss
							best_acc = max_acc
						elif min_loss == best_loss and max_acc > best_acc:
							best_trial = trial
							best_acc = max_acc

						continue

					sess.run(tf.global_variables_initializer())

					# Load saved weights for clinical NN
					with open(os.path.join(
							FLAGS.checkpoint_dir_vars + '\\' + str(f) + '\\' + str(nn_best_trials[f]), 'weights.npy'), "rb") as fp:
						weights_dict = pickle.load(fp)
					with open(os.path.join(
							FLAGS.checkpoint_dir_vars + '\\' + str(f) + '\\' + str(nn_best_trials[f]), 'bias.npy'),
							"rb") as fp:
						bias_dict = pickle.load(fp)
					for i,(key, value) in enumerate(weights_dict.items()):
						sess.run(nn_weights[key].assign(value))
					for i, (key, value) in enumerate(bias_dict.items()):
						sess.run(nn_biases[key].assign(value))

					# Load saved weights for CNN
					with open(os.path.join(
							FLAGS.checkpoint_dir_cnn + '\\' + str(f) + '\\' + str(cnn_best_trials[f]), 'weights.npy'), "rb") as fp:
						weights_dict = pickle.load(fp)
					for i,(key, value) in enumerate(weights_dict.items()):
						sess.run(cnn_model.variable_dict[key].assign(value))

					weights_cnn = sess.run(cnn_model.variable_dict)
					weights_nn = sess.run(nn_weights)

					# Init writers
					train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + str(f) + '/' + str(trial), sess.graph)
					test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + str(f) + '/' + str(trial))

					max_acc = 0
					min_loss = 1000.0
					# ==== TRAIN MODEL TILL max_epochs ====
					print('Training model...')
					for i in range(FLAGS.max_epochs):
						avg_acc_i = 0
						avg_loss_i = 0
						avg_l2_i = 0
						avg_entropy_i = 0
						# ------------ TRAIN -------------
						for step in range(training_steps):
							xi, vi, yi = dataset.Training.next_batch(FLAGS.batch_size, only_vars=False, bases3d=FLAGS.bases3d)
							if step == training_steps - 1 and i % FLAGS.print_freq == 0:
								_, summaries, acc_s, loss_s, l2_s, entropy_s = sess.run([train_op, summary_op,
																						 combined_accuracy, combined_loss, l2_loss, entropy],
									feed_dict={image_batch: xi, label_batch: yi, x_vars: vi,
											   	is_training_comb: True, is_refine: i > FLAGS.refine})
							else:
								_, acc_s, loss_s, l2_s, entropy_s = sess.run([train_op, combined_accuracy, combined_loss, l2_loss, entropy],
									feed_dict={image_batch: xi, label_batch: yi, x_vars: vi,
												is_training_comb: True, is_refine: i > FLAGS.refine})

							assert not np.isnan(loss_s), 'Model diverged with loss = NaN'

							avg_acc_i += (acc_s / training_steps)
							avg_loss_i += (loss_s / training_steps)
							avg_l2_i += (l2_s / training_steps)
							avg_entropy_i += (entropy_s / training_steps)

						# ------------ PRINT -------------
						if i % FLAGS.print_freq == 0:
							summary = tf.Summary()
							summary.value.add(tag="Accuracy", simple_value=avg_acc_i)
							summary.value.add(tag="Total_Loss", simple_value=avg_loss_i)
							summary.value.add(tag="L2_Loss", simple_value=avg_l2_i)
							summary.value.add(tag="X-entropy_Loss", simple_value=avg_entropy_i)

							train_writer.add_summary(summary, i)
							train_writer.add_summary(summaries, i)

						weights_cnn = sess.run(cnn_model.variable_dict)
						weights_nn = sess.run(nn_weights)

						# ------------ VALIDATON -------------
						if i % FLAGS.eval_freq == 0 or i == FLAGS.max_epochs-1:
							tot_acc = 0.0
							tot_loss = 0.0
							tot_l2 = 0.0
							tot_entropy = 0.0
							for step in range(validation_steps):
								xi, vi, yi = dataset.Validation.next_batch(FLAGS.batch_size, only_vars=False, bases3d=FLAGS.bases3d)
								acc_s, loss_s, l2_s, entropy_s, softmax = sess.run([combined_accuracy, combined_loss, l2_loss, entropy, combined_scores],
										feed_dict={image_batch: xi, label_batch: yi, x_vars: vi,
												   	is_training_comb: False, is_refine: i > FLAGS.refine})

								tot_acc += (acc_s / validation_steps)
								tot_loss += (loss_s / validation_steps)
								tot_l2 += (l2_s / validation_steps)
								tot_entropy += (entropy_s / validation_steps)

							summary = tf.Summary()
							summary.value.add(tag="Accuracy", simple_value=tot_acc)
							summary.value.add(tag="Total_Loss", simple_value=tot_loss)
							summary.value.add(tag="L2_Loss", simple_value=tot_l2)
							summary.value.add(tag="X-entropy_Loss", simple_value=tot_entropy)

							test_writer.add_summary(summary, i)

							# Save bets model so far
							if tot_entropy < min_loss:# and i > FLAGS.max_epochs * 0.5:
								max_acc = tot_acc
								min_loss = tot_loss

								if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial))):
									tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial)))
								checkpoint_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial) + '/' + 'best_model'
								saver.save(sess, checkpoint_path)
							print('Validation loss-acc at step %s: %s - %s' % (i, tot_loss, tot_acc))
							print('Training loss-acc at step %s: %s - %s' % (i, avg_loss_i, avg_acc_i))

					softmax_whole = []
					labels_whole = []
					test_acc = 0.0
					# Get predictions for the whole test-set
					for step in range(test_steps):
						xi, vi, yi = dataset.Test.next_batch(FLAGS.batch_size, only_vars=False, bases3d=FLAGS.bases3d)
						acc_s, softmax = sess.run([combined_accuracy, combined_scores],
												  feed_dict={image_batch: xi, label_batch: yi, x_vars: vi,
															 is_training_comb: False, is_refine: False})

						softmax_whole.append(softmax)
						labels_whole.append(yi)
						test_acc += (acc_s / test_steps)
					softmax_whole = np.reshape(softmax_whole,
											   (np.shape(softmax_whole)[0] * np.shape(softmax_whole)[1], 2))
					labels_whole = np.reshape(labels_whole, (np.shape(labels_whole)[0] * np.shape(labels_whole)[1], 2))

					# Compute ROC curve and ROC area for each class
					fpr, tpr, _ = roc_curve(labels_whole[:, 1], softmax_whole[:, 1])
					auc_k = auc(fpr, tpr)

					if not tf.gfile.Exists(FLAGS.stat_dir):
						tf.gfile.MakeDirs(FLAGS.stat_dir)
					np.save(FLAGS.stat_dir + '/min_loss_' + str(f) + '_' + str(trial) + '.npy', min_loss)
					np.save(FLAGS.stat_dir + '/max_acc_' + str(f) + '_' + str(trial) + '.npy', max_acc)
					if min_loss < best_loss:
						best_trial = trial
						best_loss = min_loss
						best_acc = max_acc
					elif min_loss == best_loss and max_acc > best_acc:
						best_trial = trial
						best_acc = max_acc

					train_writer.close()
					test_writer.close()

					dataset.Training.reset()
					dataset.Validation.reset()

				best_trials.append(best_trial)
				dataset.next_fold()

			dataset.reset()
			cv_acc_list = []
			cv_loss_list = []
			cv_auc_list = []
			mean_fpr = np.linspace(0, 1, 100)
			cv_tprs_list = []
			for f in range(FLAGS.xvalidation_folds):
				# ===== LOAD BEST MODELS FOR ALL FOLDS AND COMPUTE STATISTICS FOR TEST SET =====
				model_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(best_trials[f])
				new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'best_model.meta'))
				new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

				# ======== ROC Analysis ==========
				iters=10
				aucs = []
				accs = []
				tprs = []
				losses = []
				for k in range(iters):
					softmax_whole = []
					labels_whole = []
					test_acc = 0.0
					test_loss = 0.0
					# Get predictions for the whole test-set
					for step in range(test_steps):
						xi, vi, yi = dataset.Test.next_batch(FLAGS.batch_size, only_vars=False, bases3d=FLAGS.bases3d)
						acc_s, softmax, entropy_s = sess.run([combined_accuracy, combined_scores, entropy],
								feed_dict={image_batch: xi, label_batch: yi, x_vars: vi,
										   	is_training_comb: False, is_refine: False})

						softmax_whole.append(softmax)
						labels_whole.append(yi)
						test_acc += (acc_s / test_steps)
						test_loss += (entropy_s / test_steps)

					softmax_whole = np.reshape(softmax_whole, (np.shape(softmax_whole)[0]*np.shape(softmax_whole)[1], 2))
					labels_whole = np.reshape(labels_whole, (np.shape(labels_whole)[0]*np.shape(labels_whole)[1], 2))

					# Compute ROC curve and ROC area for each class
					fpr, tpr, _ = roc_curve(labels_whole[:,1], softmax_whole[:,1])
					tprs.append(interp(mean_fpr, fpr, tpr))
					tprs[-1][0] = 0.0

					auc_k = auc(fpr, tpr)
					aucs.append(auc_k)
					accs.append(test_acc)
					losses.append(test_loss)

				cv_auc_list.append(np.mean(aucs))
				cv_acc_list.append(np.mean(accs))
				cv_loss_list.append(np.mean(losses))
				cv_tprs_list.append(np.mean(tprs, axis=0))

				dataset.next_fold()

		avg_auc = np.mean(np.array(cv_auc_list))
		std_auc = np.std(np.array(cv_auc_list))

		avg_acc = np.mean(np.array(cv_acc_list))
		std_acc = np.std(np.array(cv_acc_list))

		avg_loss = np.mean(np.array(cv_loss_list))

		avg_tpr = np.mean(np.array(cv_tprs_list), axis=0)
		avg_tpr[-1] = 1.0
		std_tpr = np.std(np.array(cv_tprs_list), axis=0)

	#				plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
	#						 label='Luck', alpha=.8)

	#				plt.plot(mean_fpr, avg_tpr, color='b',
	#						 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (avg_auc, std_auc),
	#						 lw=2, alpha=.8)

	#				tprs_upper = np.minimum(avg_tpr + std_tpr, 1)
	#				tprs_lower = np.maximum(avg_tpr - std_tpr, 0)
	#				plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
	#								 label=r'$\pm$ 1 std. dev.')

	#				plt.xlim([-0.05, 1.05])
	#				plt.ylim([-0.05, 1.05])
	#				plt.xlabel('False Positive Rate')
	#				plt.ylabel('True Positive Rate')
	#				plt.title('Receiver operating characteristic example')
	#				plt.legend(loc="lower right")
	#				plt.show()

		# Save statistics
		if not tf.gfile.Exists(FLAGS.stat_dir):
			tf.gfile.MakeDirs(FLAGS.stat_dir)
		np.save(os.path.join(FLAGS.stat_dir + '/', 'cv_tprs_list.npy'), np.array(cv_tprs_list))
		np.save(os.path.join(FLAGS.stat_dir + '/', 'mean_fpr.npy'), mean_fpr)
		np.savetxt(os.path.join(FLAGS.stat_dir + '/', 'cv_auc_list.csv'), np.array(cv_auc_list), delimiter=",", fmt='%.5ef')
		np.savetxt(os.path.join(FLAGS.stat_dir + '/', 'cv_acc_list.csv'), np.array(cv_acc_list), delimiter=",", fmt='%.5ef')
		np.savetxt(os.path.join(FLAGS.stat_dir + '/', 'cv_loss_list.csv'), np.array(cv_loss_list), delimiter=",",
				   fmt='%.5ef')
		print('Acc/std/AUC/std/loss: %s/%s/%s/%s/%s' % (avg_acc, std_acc, avg_auc, std_auc, avg_loss))

