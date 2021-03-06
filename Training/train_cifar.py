
from cifar import cifar10_utils

from Utils import utils
from Utils import utils_final
from Models.densenet import DenseNetCifar
from Models.RFNN_densenet import RFNNDenseNetCifar

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


def train_cifar(FLAGS, NUM_GPUS):
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):
			global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

			# ====== LOAD DATASET ======
			print('Loading Dataset...')
			cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.cifar_path, training_size=FLAGS.training_size, validation_size=5000)
			print('Loading Dataset...done.')

			# ====== DEFINE SPACEHOLDERS ======
			with tf.name_scope('input'):
				if FLAGS.bases3d:
					image_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1],
												 name='x-input')
				else:
					image_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 3], name='x-input')
				label_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, 10], name='y-input')
			with tf.name_scope('alg-parameters'):
				is_training = tf.placeholder(tf.bool, name='is-training')
				weight_decay = tf.placeholder(tf.float32, shape=[], name='weight_decay')
				learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

			# ====== DEFINE FEED_DICTIONARY ======
			def feed_dict(flag):
				xs = []
				ys = []
				if flag == 0:
					for i in np.arange(NUM_GPUS):
						xi, yi = cifar10_dataset.train.next_batch(FLAGS.batch_size)
						xs.append(xi)
						ys.append(yi)
				elif flag == 1:
					for i in np.arange(NUM_GPUS):
						xi, yi = cifar10_dataset.validation.next_batch(FLAGS.batch_size)
						xs.append(xi)
						ys.append(yi)
				elif flag == 2:
					for i in np.arange(NUM_GPUS):
						xi, yi = cifar10_dataset.test.next_batch(FLAGS.batch_size)
						xs.append(xi)
						ys.append(yi)
				return {image_batch: xs, label_batch: ys, is_training: flag == 0}

			# ====== MODEL DEFINITION ======
			print('Defining model...')
			init_sigmas = [float(x) for x in FLAGS.init_sigmas.split(',')]
			comp_sigmas = [float(x) for x in FLAGS.comp_sigmas.split(',')]
			thetas = [float(x) for x in FLAGS.thetas.split(',')]

			if FLAGS.rfnn == "cnn":
				model = DenseNetCifar(
					growth_rate=FLAGS.growth_rate,
					depth=FLAGS.depth,
					total_blocks=FLAGS.total_blocks,
					keep_prob=FLAGS.keep_prob,
					model_type=FLAGS.model_type,
					is_training=is_training,
					init_kernel=FLAGS.init_kernel,
					comp_kernel=FLAGS.comp_kernel,
					reduction=FLAGS.reduction,
					bc_mode=FLAGS.bc_mode,
					avgpool_kernel_ratio=1.0,
					avgpool_stride_ratio=1.0,
					n_classes=10
				)
			else:
				model = RFNNDenseNetCifar(
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
					reduction=FLAGS.reduction,
					bc_mode=FLAGS.bc_mode,
					avgpool_kernel_ratio=1.0,
					avgpool_stride_ratio=1.0,
					n_classes=10
				)
			print('Defining model...done.')

			# ====== DEFINE LOSS, ACCURACY TENSORS ======
			print('Defining necessary OPs...')
			opt = tf.train.MomentumOptimizer(
				learning_rate, FLAGS.nesterov_momentum, use_nesterov=True)

			# === DEFINE QUEUE OPS ===
			batch_queue = tf.FIFOQueue(
				capacity=NUM_GPUS,
				dtypes=[tf.float32, tf.float32],
				shapes=[(FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1) if FLAGS.bases3d else (FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 3),
						(FLAGS.batch_size, 10)]
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

						accuracy, _, scores = tower_accuracy(logits, y)

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

		print('Training model...')
		with tf.Session(config=config) as sess:
			# Create a coordinator
			coord = tf.train.Coordinator()

			try:
				# ========= CROSS-VALIDATE RESULTS ==========
				NUM_GPUS = 1

				training_steps = int(cifar10_dataset.train.num_examples / (NUM_GPUS * FLAGS.batch_size))
				validation_steps = int(cifar10_dataset.validation.num_examples / (NUM_GPUS * FLAGS.batch_size))
				test_steps = int(cifar10_dataset.test.num_examples / (NUM_GPUS * FLAGS.batch_size))

				# ======== X TRIALS =========
				best_trial = 0
				best_loss = 1000.0
				best_acc = 0.0
#					FLAGS.trials = 1  # TEST!!!!!!!!!!!!!!!!!!!!!!
				for trial in range(FLAGS.trials):
					print('Trial: ' + str(trial))

					# Check if trial already run
					if tf.gfile.Exists(FLAGS.stat_dir + '/min_loss_' + '_' + str(trial) + '.npy'):
						min_loss = np.load(FLAGS.stat_dir + '/min_loss_' + '_' + str(trial) + '.npy')
						max_acc = np.load(FLAGS.stat_dir + '/max_acc_' + '_' + str(trial) + '.npy')

						if min_loss < best_loss:
							best_trial = trial
							best_loss = min_loss
							best_acc = max_acc
						elif min_loss == best_loss and max_acc > best_acc:
							best_trial = trial
							best_acc = max_acc

						continue

					lr = FLAGS.learning_rate

					sess.run(tf.global_variables_initializer())

					# Save initial kernels from 1st layer
					kernels_tensor = model.kernels[0]
					kernels = sess.run(kernels_tensor)
					kernels = np.mean(kernels, axis=2)

					if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial))):
						tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial)))
					checkpoint_path = FLAGS.checkpoint_dir + '/' + '/' + str(trial) + '/' + 'kernels_init.npy'
					np.save(checkpoint_path, kernels)
#						show_kernels(kernels)

					# Launch the queue runner threads.
					tf.train.start_queue_runners(sess, coord=coord)

					# Init writers
					train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + '/' + str(trial), sess.graph)
					test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + '/' + str(trial))

					# ======== LSUV INIT WEIGHTS WITH A FORWARD PASS ==========
					xs = []
					ys = []
					print("Dequeue data for init...")
					for i in range(FLAGS.t_max):
						sess.run(batch_enqueue, feed_dict=feed_dict(0))
						batch, label = sess.run([x,y], feed_dict=feed_dict(0))
						xs.append(batch)
						ys.append(label)
					print("Dequeue data for init...done.")

					if FLAGS.rfnn != "cnn":
						print("Initializing weights with LSUV...")
						for l in range(len(model.alphas)):
							var = 0.0
							t_i = 0
							while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
								alphas, b_l = sess.run([model.alphas[l], model.conv_act[l]], feed_dict={x: xs[t_i], is_training: False})
								var = np.var(b_l)
								sess.run(model.alphas[l].assign(alphas / np.sqrt(var)))
								t_i += 1
						print("Initializing weights with LSUV...done.")

					# ======== INIT WEIGHT DECAY ==========
#					print("Initializing weight decay...")
#					FLAGS.weight_decay = 0.0
#					for i in range(FLAGS.t_max):
#						x_loss, l2_loss = sess.run([avg_loss_entropy, avg_loss_l2],
#												   feed_dict={x: xs[i], y: ys[i], weight_decay: 1.0, is_training: False})
#						FLAGS.weight_decay += x_loss/l2_loss/FLAGS.t_max
#					print("Initializing weight decay done....")

					max_acc = 0
					min_loss = 1000.0
					# ==== TRAIN MODEL TILL max_epochs ====
#						FLAGS.max_epochs = 1  # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					for i in range(FLAGS.max_epochs):
						if coord.should_stop():
							break

						# ----- Reduce learning rate / increase bnorm momentum ------
						if i == FLAGS.reduce_lr_epoch_1:
							lr = lr /10
						if i == FLAGS.reduce_lr_epoch_2:
							lr = lr /10

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
											feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, learning_rate: lr})
							else:
								_, acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s\
									= sess.run([train_op, avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
											feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, learning_rate: lr})

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
										feed_dict={is_training: False, weight_decay: FLAGS.weight_decay})

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

								if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial))):
									tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial)))
								checkpoint_path = FLAGS.checkpoint_dir + '/' + str(trial) + '/' + 'best_model'
								saver.save(sess, checkpoint_path)
							print('Validation loss-acc at step %s: %s - %s' % (i, tot_loss, tot_acc))

					if not tf.gfile.Exists(FLAGS.stat_dir):
						tf.gfile.MakeDirs(FLAGS.stat_dir)
					np.save(FLAGS.stat_dir + '/min_loss_' + '_' + str(trial) + '.npy', min_loss)
					np.save(FLAGS.stat_dir + '/max_acc_' + '_' + str(trial) + '.npy', max_acc)
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

					if not tf.gfile.Exists(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial))):
						tf.gfile.MakeDirs(os.path.join(FLAGS.checkpoint_dir + '/' + str(trial)))
					checkpoint_path = FLAGS.checkpoint_dir + '/' + '/' + str(
						trial) + '/' + 'kernels_trained.npy'
					np.save(checkpoint_path, kernels)
#						show_kernels(kernels)

					cifar10_dataset.train.reset()
					cifar10_dataset.validation.reset()
					cifar10_dataset.test.reset()

#				best_trial = 0
				# ===== LOAD BEST MODELS FOR ALL FOLDS AND COMPUTE STATISTICS FOR TEST SET =====
				model_path = FLAGS.checkpoint_dir + '/' + str(best_trial)
				new_saver = tf.train.import_meta_graph(os.path.join(model_path, 'best_model.meta'))
				new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

				iters=10
				accs = []
				losses = []
				for k in range(iters):
					test_acc = 0.0
					test_loss = 0.0
					# Get predictions for the whole test-set
					for step in range(test_steps):
						sess.run(batch_enqueue, feed_dict=feed_dict(2))
						acc_s, softmax, labels, entropy = sess.run([avg_accuracy, scores, y, avg_loss_entropy], feed_dict={is_training: False})

						test_acc += (acc_s / test_steps)
						test_loss += (entropy / test_steps)
					accs.append(test_acc)
					losses.append(test_loss)

				avg_acc = np.mean(np.array(accs))
				avg_loss = np.mean(np.array(losses))
				std_acc = np.std(np.array(accs))

				# Save statistics
				if not tf.gfile.Exists(FLAGS.stat_dir):
					tf.gfile.MakeDirs(FLAGS.stat_dir)
				np.savetxt(os.path.join(FLAGS.stat_dir + '/', 'cv_acc_list.csv'), np.array(accs), delimiter=",", fmt='%.5ef')
				np.savetxt(os.path.join(FLAGS.stat_dir + '/', 'cv_loss_list.csv'), np.array(losses), delimiter=",",
						   fmt='%.5ef')

				print('Acc/std : %s/%s/%s' % (avg_acc, std_acc, avg_loss))

			except Exception as e:
				# Report exceptions to the coordinator.
				coord.request_stop(e)
			finally:
				# Terminate threads
				coord.request_stop()
				coord.join()
			sess.run(close_queue)

