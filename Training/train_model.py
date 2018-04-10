
from Utils import utils
from Utils import utils_final
from Models.densenet import DenseNet
from Models.densenet3d import DenseNet3d
from Models.RFNN_densenet import RFNNDenseNet
from Models.RFNN_densenet3d import RFNNDenseNet3D

from Utils.training_utils import average_gradients
from Utils.training_utils import tower_loss_dense
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

import gradient_checkpointing_master.memory_saving_gradients as memory_saving_gradients
from tensorflow.python.ops import gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', gate_gradients=True, **kwargs)
gradients.__dict__["gradients"] = gradients_memory


def train_ctnet(FLAGS, NUM_GPUS):
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

			# ====== LOAD DATASET ======
			print('Loading Dataset...')
			training_points, test_points, validation_points = utils_final.read_dataset(FLAGS.datapath)

			dataset = utils_final.DataSet(training_points, test_points, validation_points,
									normalize=FLAGS.normalization, img3d=FLAGS.bases3d)
			print('Loading Dataset...done.')

			# ====== DEFINE SPACEHOLDERS ======
			with tf.name_scope('input'):
				if FLAGS.bases3d:
					image_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1],
												 name='x-input')
				else:
					image_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 1], name='x-input')
				label_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, 2], name='y-input')
			with tf.name_scope('alg-parameters'):
				is_training = tf.placeholder(tf.bool, name='is-training')
				weight_decay = tf.placeholder(tf.float32, shape=[], name='weight_decay')
				bnorm_momentum = tf.placeholder(tf.float32, shape=[], name='bnorm_momentum')

			# ====== DEFINE FEED_DICTIONARY ======
			def feed_dict(flag):
				xs = []
				ys = []
				if flag == 0:
					for i in np.arange(NUM_GPUS):
						xi, yi = dataset.Training.next_batch(FLAGS.batch_size, bases3d=FLAGS.bases3d)
						xs.append(xi)
						ys.append(yi)
				elif flag == 1:
					for i in np.arange(NUM_GPUS):
						xi, yi = dataset.Validation.next_batch(FLAGS.batch_size, bases3d=FLAGS.bases3d)
						xs.append(xi)
						ys.append(yi)
				elif flag == 2:
					for i in np.arange(NUM_GPUS):
						xi, yi = dataset.Test.next_batch(FLAGS.batch_size, bases3d=FLAGS.bases3d)
						xs.append(xi)
						ys.append(yi)
				return {image_batch: xs, label_batch: ys, is_training: flag == 0}

			# ====== MODEL DEFINITION ======
			print('Defining model...')
			init_sigmas = [float(x) for x in FLAGS.init_sigmas.split(',')]
			comp_sigmas = [float(x) for x in FLAGS.comp_sigmas.split(',')]
			thetas = [float(x) for x in FLAGS.thetas.split(',')]
			phis = [float(x) for x in FLAGS.phis.split(',')]

			if FLAGS.rfnn == "cnn":
				if FLAGS.bases3d:
					model = DenseNet3d(
						growth_rate=FLAGS.growth_rate,
						depth=FLAGS.depth,
						total_blocks=FLAGS.total_blocks,
						keep_prob=FLAGS.keep_prob,
						model_type=FLAGS.model_type,
						is_training=is_training,
						init_kernel=FLAGS.init_kernel,
						comp_kernel=FLAGS.comp_kernel,
						bnorm_momentum=FLAGS.bnorm_mom,
						renorm=FLAGS.renorm,
						beta_wd=FLAGS.beta_wd,
						reduction=FLAGS.reduction,
						bc_mode=FLAGS.bc_mode,
						n_classes=2)
				else:
					model = DenseNet(
						growth_rate=FLAGS.growth_rate,
						depth=FLAGS.depth,
						total_blocks=FLAGS.total_blocks,
						keep_prob=FLAGS.keep_prob,
						model_type=FLAGS.model_type,
						is_training=is_training,
						init_kernel=FLAGS.init_kernel,
						comp_kernel=FLAGS.comp_kernel,
						bnorm_momentum=FLAGS.bnorm_mom,
						renorm=FLAGS.renorm,
						beta_wd=FLAGS.beta_wd,
						reduction=FLAGS.reduction,
						bc_mode=FLAGS.bc_mode,
						n_classes=2)
			else:
				if FLAGS.bases3d:
					model = RFNNDenseNet3D(
						growth_rate=FLAGS.growth_rate,
						depth=FLAGS.depth,
						total_blocks=FLAGS.total_blocks,
						keep_prob=FLAGS.keep_prob,
						model_type=FLAGS.model_type,
						is_training=is_training,
						init_kernel=FLAGS.init_kernel,
						comp_kernel=FLAGS.comp_kernel,
						init_sigmas=init_sigmas,
						comp_sigmas=comp_sigmas,
						init_order=FLAGS.init_order,
						comp_order=FLAGS.comp_order,
						thetas=thetas,
						phis=phis,
						rfnn=FLAGS.rfnn,
						bnorm_momentum=FLAGS.bnorm_mom,
						renorm=FLAGS.renorm,
						beta_wd=FLAGS.beta_wd,
						reduction=FLAGS.reduction,
						bc_mode=FLAGS.bc_mode,
						n_classes=2)
				else:
					model = RFNNDenseNet(
						growth_rate=FLAGS.growth_rate,
						depth=FLAGS.depth,
						total_blocks=FLAGS.total_blocks,
						keep_prob=FLAGS.keep_prob,
						model_type=FLAGS.model_type,
						is_training=is_training,
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

			# ====== DEFINE LOSS, ACCURACY TENSORS ======
			print('Defining necessary OPs...')
			opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon)

			# === DEFINE QUEUE OPS ===
			batch_queue = tf.FIFOQueue(
				capacity=NUM_GPUS,
				dtypes=[tf.float32, tf.float32],
				shapes=[(FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1) if FLAGS.bases3d else (FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 1),
						(FLAGS.batch_size, 2)]
			)
			batch_enqueue = batch_queue.enqueue_many([image_batch, label_batch])
			close_queue = batch_queue.close()

		# Init lists
		tower_grads = []
		tower_losses = []
		tower_losses_l2 = []
		tower_losses_entropy = []
		tower_accuracies = []

		# ====== FORWARD PASS ======
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(NUM_GPUS):
				x, y = batch_queue.dequeue()

				with tf.device('/gpu:%d' % i):
					# ====== INFERENCE ======
					with tf.name_scope('%s_%d' % ('tower', i)) as scope:
						# Calculate predictions
						logits = model.inference(x)
						x_entropy, l2 = tower_loss_dense(logits, y)

						l2_loss = l2*weight_decay
						tf.add_to_collection('losses', l2_loss)
						tf.add_to_collection('losses', x_entropy)
						with tf.name_scope('Total_Loss'):
							loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')

						accuracy, _, scores = tower_accuracy(logits, y, scope)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						# Calculate the gradients for the batch of data on this tower.
						grads = tf.gradients(loss, tf.trainable_variables())
						grads_and_vars = list(zip(grads, tf.trainable_variables()))

						# Keep track of the gradients across all towers.
						tower_grads.append(grads_and_vars)
						tower_losses.append(loss)
						tower_losses_entropy.append(x_entropy)
						tower_losses_l2.append(l2_loss)
						tower_accuracies.append(accuracy)
		# ====== BACKWARD PASS ======
		with tf.device('/cpu:0'):
			# Calculate the mean of each gradient - synchronization point across towers.
			avg_grads = average_gradients(tower_grads)
			avg_loss = tf.reduce_mean(tower_losses, 0)
			avg_loss_entropy = tf.reduce_mean(tower_losses_entropy, 0)
			avg_loss_l2 = tf.reduce_mean(tower_losses_l2, 0)
			avg_accuracy = tf.reduce_mean(tower_accuracies, 0)
			print('Defining necessary OPs...done.')

			# ====== ADD SUMMARIES ======
			# Gradients
			for grad, var in avg_grads:
				if grad is not None:
					summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
			# Trainable variables
			for var in tf.trainable_variables():
				summaries.append(tf.summary.histogram(var.op.name, var))
			# RFNN kernels
			for kernel in model.kernels:
				summaries.append(tf.summary.histogram(kernel.op.name, kernel))

			# Print initial kernels
			# alphas_tensor, kernels_tensor = get_kernels()
			# alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])
			# np.save('./Kernels/kernel_0.npy', kernels_array)
			# np.save('./Kernels/alphas_0.npy', alphas)

			# ====== UPDATE VARIABLES ======
			print('Defining update OPs...')
			# Apply the gradients to adjust the shared variables.
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = opt.apply_gradients(avg_grads, global_step=global_step)
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
			# Create a coordinator
			coord = tf.train.Coordinator()

			try:
				# ========= CROSS-VALIDATE RESULTS ==========
				NUM_GPUS = 1
				best_trials = []
#				FLAGS.xvalidation_folds = 1  # TEST!!!!!!!!!!!!!!!!!!!!
				for f in range(FLAGS.xvalidation_folds):
					print('X-val fold: ' + str(f))
					training_steps = int(np.ceil(2 * dataset.Training.num_examples / (NUM_GPUS * FLAGS.batch_size)))
					validation_steps = int(np.ceil(2 * dataset.Validation.num_examples / (NUM_GPUS * FLAGS.batch_size)))
					bnorm_mom = FLAGS.bnorm_mom

					# ======== X TRIALS =========
					best_trial = 0
					best_loss = 1000.0
					best_acc = 0.0
#					FLAGS.trials = 1  # TEST!!!!!!!!!!!!!!!!!!!!!!
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

						# Save initial kernels from 1st layer
						kernels_tensor = model.kernels[0]
						kernels = sess.run(kernels_tensor)
						kernels = np.mean(kernels, axis=2)

						if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial))):
							tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial)))
						checkpoint_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial) + '/' + 'kernels_init.npy'
						np.save(checkpoint_path, kernels)
#						show_kernels(kernels)

						# Launch the queue runner threads.
						tf.train.start_queue_runners(sess, coord=coord)

						# Init writers
						train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + str(f) + '/' + str(trial), sess.graph)
						test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + str(f) + '/' + str(trial))

						# ======== LSUV INIT WEIGHTS WITH A FORWARD PASS ==========
						xs = []
						ys = []
						if not FLAGS.bases3d:
							print("Dequeue data for init...")
							for i in range(FLAGS.t_max):
								sess.run(batch_enqueue, feed_dict=feed_dict(0))
								batch, label = sess.run([x,y], feed_dict=feed_dict(0))
								xs.append(batch)
								ys.append(label)
							print("Dequeue data for init...done.")

							print("Initializing weights with LSUV...")
							for l in range(len(model.alphas)):
								var = 0.0
								t_i = 0
								while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
									alphas, b_l = sess.run([model.alphas[l], model.conv_act[l]], feed_dict={x: xs[t_i], is_training: False, bnorm_momentum: bnorm_mom})
									var = np.var(b_l)
									sess.run(model.alphas[l].assign(alphas / np.sqrt(var)))
									t_i += 1
							print("Initializing weights with LSUV...done.")

							# ======== INIT WEIGHT DECAY ==========
							print("Initializing weight decay...")
							FLAGS.weight_decay = 0.0
							for i in range(FLAGS.t_max):
								x_loss, l2_loss = sess.run([avg_loss_entropy, avg_loss_l2],
														   feed_dict={x: xs[i], y: ys[i], weight_decay: 1.0, is_training: False,
																	  bnorm_momentum: bnorm_mom})
								FLAGS.weight_decay += x_loss/l2_loss/FLAGS.t_max
							print("Initializing weight decay done....")
						else:
							print("Initializing weights with LSUV...")
#							for l in range(len(model.alphas)):
#								var = 0.0
#								t_i = 0
#								while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
#									sess.run(batch_enqueue, feed_dict=feed_dict(0))
#									alphas, b_l = sess.run([model.alphas[l], model.conv_act[l]],
#														   feed_dict={is_training: False,
#																	  bnorm_momentum: bnorm_mom})
#									var = np.var(b_l)
#									sess.run(model.alphas[l].assign(alphas / np.sqrt(var)))
#									t_i += 1
							print("Initializing weights with LSUV...done.")

							# ======== INIT WEIGHT DECAY ==========
							print("Initializing weight decay...")
							FLAGS.weight_decay = 0.0
							for i in range(FLAGS.t_max):
								sess.run(batch_enqueue, feed_dict=feed_dict(0))
								x_loss, l2_loss = sess.run([avg_loss_entropy, avg_loss_l2],
														   feed_dict={weight_decay: 1.0,
																	  is_training: False,
																	  bnorm_momentum: bnorm_mom})
								FLAGS.weight_decay += x_loss / l2_loss / FLAGS.t_max
							print("Initializing weight decay done....")

						max_acc = 0
						min_loss = 1000.0
						# ==== TRAIN MODEL TILL max_epochs ====
#						FLAGS.max_epochs = 1  # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						print('Training model...')
						for i in range(FLAGS.max_epochs):
							if coord.should_stop():
								break

							# ----- Reduce learning rate / increase bnorm momentum ------
							if i == FLAGS.reduce_lr_epoch_1:
								if FLAGS.bnorm_inc:
									bnorm_mom = bnorm_mom * 1.0714
							if i == FLAGS.reduce_lr_epoch_2:
								if FLAGS.bnorm_inc:
									bnorm_mom = bnorm_mom * 1.0714

							avg_acc_i = 0
							avg_loss_i = 0
							avg_xentropy_loss_i = 0
							avg_l2_loss_i = 0
							# ------------ TRAIN -------------
							for step in range(training_steps):
								sess.run(batch_enqueue, feed_dict=feed_dict(0))
								if step == training_steps - 1 and i % FLAGS.print_freq == 0:
									_, summaries, acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s \
										= sess.run([train_op, summary_op, avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
												feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, bnorm_momentum: bnorm_mom})
								else:
									_, acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s\
										= sess.run([train_op, avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
												feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, bnorm_momentum: bnorm_mom})

								assert not np.isnan(loss_s), 'Model diverged with loss = NaN'

								avg_acc_i += (acc_s / training_steps)
								avg_loss_i += (loss_s / training_steps)
								avg_l2_loss_i += (l2_loss_value_s / training_steps)
								avg_xentropy_loss_i += (xentropy_loss_value_s / training_steps)

							# ------------ PRINT -------------
							if i % FLAGS.print_freq == 0:
								summary = tf.Summary()
								summary.value.add(tag="Accuracy", simple_value=avg_acc_i)
								summary.value.add(tag="Total_Loss", simple_value=avg_loss_i)
								summary.value.add(tag="X-entropy_Loss", simple_value=avg_xentropy_loss_i)
								summary.value.add(tag="L2_Loss", simple_value=avg_l2_loss_i)

								train_writer.add_summary(summaries, i)
								train_writer.add_summary(summary, i)

							# ------------ VALIDATON -------------
							if i % FLAGS.eval_freq == 0 or i == FLAGS.max_epochs-1:
								tot_acc = 0.0
								tot_loss = 0.0
								tot_loss_l2 = 0.0
								tot_loss_entropy = 0.0
								for step in range(validation_steps):
									sess.run(batch_enqueue, feed_dict=feed_dict(1))
									acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s\
										= sess.run([avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
											feed_dict={is_training: False, weight_decay: FLAGS.weight_decay, bnorm_momentum: bnorm_mom})

									tot_acc += (acc_s / validation_steps)
									tot_loss += (loss_s / validation_steps)
									tot_loss_l2 += (l2_loss_value_s / validation_steps)
									tot_loss_entropy += (xentropy_loss_value_s / validation_steps)

								summary = tf.Summary()
								summary.value.add(tag="Accuracy", simple_value=tot_acc)
								summary.value.add(tag="Total_Loss", simple_value=tot_loss)
								summary.value.add(tag="X-entropy_Loss", simple_value=tot_loss_entropy)
								summary.value.add(tag="L2_Loss", simple_value=tot_loss_l2)

								test_writer.add_summary(summary, i)

								# Save bets model so far
								if tot_loss_entropy < min_loss: # and i > FLAGS.max_epochs * 0.6:
									max_acc = tot_acc
									min_loss = tot_loss_entropy

									if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial))):
										tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial)))
									checkpoint_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial) + '/' + 'best_model'
									saver.save(sess, checkpoint_path)
								print('Validation loss-acc at step %s: %s - %s' % (i, tot_loss, tot_acc))

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

						# Save trained kernels from 1st layer
						kernels_tensor = model.kernels[0]
						kernels = sess.run(kernels_tensor)
						kernels = np.mean(kernels, axis=2)

						if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial))):
							tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(trial)))
						checkpoint_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(
							trial) + '/' + 'kernels_trained.npy'
						np.save(checkpoint_path, kernels)
#						show_kernels(kernels)

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
				test_steps = int(np.ceil(2 * dataset.Test.num_examples / (NUM_GPUS * FLAGS.batch_size)))
				for f in range(FLAGS.xvalidation_folds):
					model_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(best_trials[f])
					new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'best_model.meta'))
					new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
#						# restore
#					for j in range(3):
#						model_path = FLAGS.checkpoint_dir + '/' + str(f) + '/' + str(j)
#						new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'best_model.meta'))
#						new_saver.restore(sess, tf.train.latest_checkpoint(model_path))
#						checkpoint_path = model_path.replace('SO', 'Shallow')
#						if not tf.gfile.Exists(checkpoint_path):
#							tf.gfile.MakeDirs(checkpoint_path)
#						saver.save(sess, checkpoint_path + '/best_model')

					# ======== ROC Analysis ==========
					iters=10
					aucs = []
					accs = []
					tprs = []
					for k in range(iters):
						softmax_whole = []
						labels_whole = []
						test_acc = 0.0
						# Get predictions for the whole test-set
						for step in range(test_steps):
							sess.run(batch_enqueue, feed_dict=feed_dict(2))
							acc_s, softmax, labels = sess.run([avg_accuracy, scores, y], feed_dict={is_training: False})

							softmax_whole.append(softmax)
							labels_whole.append(labels)
							test_acc += (acc_s / test_steps)
						softmax_whole = np.reshape(softmax_whole, (np.shape(softmax_whole)[0]*np.shape(softmax_whole)[1], 2))
						labels_whole = np.reshape(labels_whole, (np.shape(labels_whole)[0]*np.shape(labels_whole)[1], 2))

						# Compute ROC curve and ROC area for each class
						labels_whole = np.array([0 if x[0] == 1 else 1 for x in labels_whole])
						softmax_whole = np.array([score[1] for score in softmax_whole])

						fpr, tpr, _ = roc_curve(labels_whole, softmax_whole)
						tprs.append(interp(mean_fpr, fpr, tpr))
						tprs[-1][0] = 0.0

						auc_k = auc(fpr, tpr)
						aucs.append(auc_k)
						accs.append(test_acc)

					cv_auc_list.append(np.mean(aucs))
					cv_acc_list.append(np.mean(accs))
					cv_tprs_list.append(np.mean(tprs, axis=0))


#				plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#						 label='Luck', alpha=.8)

				avg_auc = np.mean(np.array(cv_auc_list))
				std_auc = np.std(np.array(cv_auc_list))

				avg_acc = np.mean(np.array(cv_acc_list))
				std_acc = np.std(np.array(cv_acc_list))

				avg_tpr = np.mean(np.array(cv_tprs_list), axis=0)
				avg_tpr[-1] = 1.0
				std_tpr = np.std(np.array(cv_tprs_list), axis=0)

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

				print('Acc/std/AUC/std : %s/%s/%s/%s' % (avg_acc, std_acc, avg_auc, std_auc))

			except Exception as e:
				# Report exceptions to the coordinator.
				coord.request_stop(e)
			finally:
				# Terminate threads
				coord.request_stop()
				coord.join()
			sess.run(close_queue)
