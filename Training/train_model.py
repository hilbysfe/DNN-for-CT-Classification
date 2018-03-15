
from Utils import utils
from Models.rfnn import RFNN
from Models.ctnet import CTNET
from Models.densenet import DenseNet
from Models.densenet3d import DenseNet3d
from Models.RFNN_densenet import RFNNDenseNet
from Models.ae import Autoencoder

from Utils.training_utils import tower_loss
from Utils.training_utils import average_gradients
from Utils.training_utils import tower_loss_dense
from Utils.training_utils import tower_accuracy

import numpy as np
import tensorflow as tf
import pickle
import os

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
			with open(FLAGS.trainingpath, 'rb') as handle:
				training_points = pickle.load(handle)
			with open(FLAGS.testpath, 'rb') as handle:
				test_points = pickle.load(handle)

			dataset = utils.DataSet(training_points, test_points,
									cross_validation_folds=FLAGS.xvalidation_folds,
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
				learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
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
				return {image_batch: xs, label_batch: ys, is_training: flag == 0}

			# ====== MODEL DEFINITION ======
			print('Defining model...')

			init_sigmas = [float(x) for x in FLAGS.init_sigmas.split(',')]
			comp_sigmas = [float(x) for x in FLAGS.comp_sigmas.split(',')]
			thetas = [float(x) for x in FLAGS.thetas.split(',')]
#			kernels = [int(x) for x in FLAGS.kernels.split(',')]
#			maps = [int(x) for x in FLAGS.maps.split(',')]
#			bases = [int(x) for x in FLAGS.bases.split(',')]
#			strides = [int(x) for x in FLAGS.strides.split(',')]

			#            model = RFNN(
			#                n_classes=2,
			#                kernels=kernels,
			#                maps=maps,
			#                sigmas=sigmas,
			#                bases=bases,
			#                bases_3d=FLAGS.bases3d,
			#                is_training=is_training,
			#                batchnorm=FLAGS.batch_normalization
			#            )

			#            model = CTNET(
			#                n_classes=2,
			#                kernels=kernels,
			#                maps=maps,
			#                strides=strides,
			#                pretraining=False,
			#                is_training = is_training,
			#                conv3d=FLAGS.bases3d,
			#                bnorm=FLAGS.batch_normalization)

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
				n_classes=2
			)

			print('Defining model...done.')

			# ====== DEFINE LOSS, ACCURACY TENSORS ======
			print('Defining necessary OPs...')

			#            opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
#			opt = tf.train.MomentumOptimizer(
#				FLAGS.learning_rate, FLAGS.nesterov_momentum, use_nesterov=True)
			opt = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon)
#			opt = tf.contrib.opt.NadamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon)

			# === DEFINE QUEUE OPS ===
			batch_queue = tf.FIFOQueue(
				capacity=NUM_GPUS,
				dtypes=[tf.float32, tf.float32],
				shapes=[(FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, FLAGS.Z_dim, 1) if FLAGS.bases3d else (FLAGS.batch_size, FLAGS.X_dim, FLAGS.X_dim, 1),
						(FLAGS.batch_size, 2)]
			)
			batch_enqueue = batch_queue.enqueue_many([image_batch, label_batch])
			close_queue = batch_queue.close()

		# Calculate the gradients for each model tower.
		tower_grads = []
		tower_losses = []
		tower_losses_l2 = []
		tower_losses_entropy = []
		tower_accuracies = []
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(NUM_GPUS):

				x, y = batch_queue.dequeue()

				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % ('tower', i)) as scope:
						# ====== INFERENCE ======
						if FLAGS.pretraining:
							print('Pre-training model...')

							network_architecture = \
								{
									'Conv_kernels': kernels,
									'Conv_maps': maps
								}
							ae = Autoencoder(network_architecture)

							assign_ops, net = ae.load_weights(x, FLAGS.pretrained_weights_path,
															  FLAGS.pretrained_biases_path, is_training)

							# Calculate predictions
							logits = model.inference(net)

							print('Pre-training model...done.')
						else:
							# Calculate predictions
							logits = model.inference(x)

						x_entropy, l2 = tower_loss_dense(logits, y)
						l2_loss = l2*weight_decay
						tf.add_to_collection('losses', l2_loss)
						tf.add_to_collection('losses', x_entropy)

						with tf.name_scope('Total_Loss'):
							loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')

						#						loss = tower_loss(logits, y, scope)
						accuracy = tower_accuracy(logits, y, scope)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						# Calculate the gradients for the batch of data on this tower.
						grads = tf.gradients(loss, tf.trainable_variables())
						grads_and_vars = list(zip(grads, tf.trainable_variables()))

#						grads = opt.compute_gradients(loss)

						# Keep track of the gradients across all towers.
						tower_grads.append(grads_and_vars)
						tower_losses.append(loss)
						tower_losses_entropy.append(x_entropy)
						tower_losses_l2.append(l2_loss)
						tower_accuracies.append(accuracy)
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

			# Track the moving averages of all trainable variables.
#			variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
#			variables_averages_op = variable_averages.apply(tf.trainable_variables())

			# Group all updates into a single train op.
#			train_op = tf.group(apply_gradient_op, variables_averages_op)

			print('Defining update OPs...done.')

			# ====== SAVING OPS ======

			# Create a saver.
			saver = tf.train.Saver(tf.global_variables())

			# Build the summary operation from the last tower summaries.
			summary_op = tf.summary.merge(summaries)

			# ====== DEFINE SESSION AND OPTIMIZE ======
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:

			print('Training model...')
			avg_val_acc = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_val_l2 = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_val_entropy = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_val_total = np.zeros(FLAGS.max_epochs, dtype=np.float32)

			avg_train_acc = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_train_l2 = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_train_entropy = np.zeros(FLAGS.max_epochs, dtype=np.float32)
			avg_train_total = np.zeros(FLAGS.max_epochs, dtype=np.float32)

			# Create a coordinator
			coord = tf.train.Coordinator()

			try:
				avg_train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/avg', sess.graph)
				avg_test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/avg')

				FLAGS.xvalidation_folds = 2
				for f in range(FLAGS.xvalidation_folds):
					training_steps = int(np.ceil(2 * dataset.Training.num_examples / (NUM_GPUS * FLAGS.batch_size)))
					validation_steps = int(np.ceil(2 * dataset.Validation.num_examples / (FLAGS.batch_size * NUM_GPUS)))
					sess.run(tf.global_variables_initializer())
					lr = FLAGS.learning_rate
					bnorm_mom = FLAGS.bnorm_mom

					#                    	print([n.name for n in tf.get_default_graph().as_graph_def().node
					#                               if 'ConvLayer1' in n.name and 'alphas' in n.name])

					# Show initial kernels from 1st layer
					#				alphas_t, kernels_t = get_kernels(1)
					#				alphas, kernels = sess.run([alphas_t, kernels_t])
					#				show_kernels(kernels)

					# Launch the queue runner threads.
					tf.train.start_queue_runners(sess, coord=coord)

					# Assign ops if pre-training
#					if FLAGS.pretraining:
#						for assign_op in assign_ops:
#							sess.run(assign_op)

					# Init writers
					train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + str(f), sess.graph)
					test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + str(f))

					# ======== LSUV INIT WEIGHTS WITH A FORWARD PASS ==========
					# Dequeue data
					xs = []
					ys = []
					for i in range(FLAGS.t_max):
						sess.run(batch_enqueue, feed_dict=feed_dict(0))
						batch, label = sess.run([x,y], feed_dict=feed_dict(0))
						xs.append(batch)
						ys.append(label)

					print("Initializing weights with LSUV...")
					# Conv layers
					for l in range(len(model.alphas)):
#						print(l)
						var = 0.0
						t_i = 0
						while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
							alphas, b_l = sess.run([model.alphas[l], model.conv_act[l]], feed_dict={x: xs[t_i], is_training: False, bnorm_momentum: bnorm_mom})
							var = np.var(b_l)
#							print(var)
							sess.run(model.alphas[l].assign(alphas / np.sqrt(var)))
							t_i += 1
#					# FCL layer
#					for l in range(len(model.weights)):
#						var = 0.0
#						t_i = 0
#						while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
#							weights, b_l = sess.run([model.weights[l], model.fl_act[l]], feed_dict={x: xs[t_i], is_training: False})
#							var = np.var(b_l)
#							print(var)
#							sess.run(model.weights[l].assign(weights / np.sqrt(var)))
#							t_i += 1
#
#					# Bottleneck layers
#					if FLAGS.bc_mode:
#						print("bottleneck")
#						for l in range(len(model.bc_conv_act)):
#							var = 0.0
#							t_i = 0
#							while (abs(var - 1.0) >= FLAGS.tol_var or t_i <= FLAGS.t_min) and t_i < FLAGS.t_max:
#								w_l, b_l = sess.run([model.bc_weights[l], model.bc_conv_act[l]],
#													feed_dict={x: xs[t_i], is_training: False})
#								var = np.var(b_l)
#								#							print(var)
#								sess.run(model.bc_weights[l].assign(w_l / np.sqrt(var)))
#								t_i += 1
					print("Initializing weights with LSUV...done.")

					print("Initializing weight decay...")
					# Dequeue data
					FLAGS.weight_decay = 0.0
					for i in range(FLAGS.t_max):
						x_loss, l2_loss = sess.run([avg_loss_entropy, avg_loss_l2],
												   feed_dict={x: xs[i], y: ys[i], weight_decay: 1.0, is_training: False,
															  bnorm_momentum: bnorm_mom})
						FLAGS.weight_decay += x_loss/l2_loss/FLAGS.t_max
					print('wd: ' + str(FLAGS.weight_decay))
					print("Initializing weight decay done....")

					max_acc = 0
					for i in range(FLAGS.max_epochs):

						if coord.should_stop():
							break

						# ----- Reduce learning rate ------
						if i == FLAGS.reduce_lr_epoch_1:
#							lr = lr / 10
							if FLAGS.bnorm_inc:
								bnorm_mom = bnorm_mom * 1.0714
						if i == FLAGS.reduce_lr_epoch_2:
#							lr = lr / 10
							if FLAGS.bnorm_inc:
								bnorm_mom = bnorm_mom * 1.0714

						# ------------ TRAIN -------------
						avg_acc_i = 0
						avg_loss_i = 0
						avg_xentropy_loss_i = 0
						avg_l2_loss_i = 0
						for step in range(training_steps):
							sess.run(batch_enqueue, feed_dict=feed_dict(0))
							if step == training_steps - 1 and i % FLAGS.print_freq == 0:
								_, summaries, acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s \
									= sess.run([train_op, summary_op, avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
											feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, learning_rate: lr, bnorm_momentum: bnorm_mom})
							else:
								_, acc_s, loss_s, l2_loss_value_s, xentropy_loss_value_s\
									= sess.run([train_op, avg_accuracy, avg_loss, avg_loss_l2, avg_loss_entropy],
											feed_dict={is_training: True, weight_decay: FLAGS.weight_decay, learning_rate: lr, bnorm_momentum: bnorm_mom})

							assert not np.isnan(loss_s), 'Model diverged with loss = NaN'

							avg_acc_i += (acc_s / training_steps)
							avg_loss_i += (loss_s / training_steps)
							avg_l2_loss_i += (l2_loss_value_s / training_steps)
							avg_xentropy_loss_i += (xentropy_loss_value_s / training_steps)

						if i % FLAGS.print_freq == 0:
							# ------------ PRINT -------------
							summary = tf.Summary()
							summary.value.add(tag="Accuracy", simple_value=avg_acc_i)
							summary.value.add(tag="Total_Loss", simple_value=avg_loss_i)
							summary.value.add(tag="X-entropy_Loss", simple_value=avg_xentropy_loss_i)
							summary.value.add(tag="L2_Loss", simple_value=avg_l2_loss_i)
							summary.value.add(tag="Learning_rate", simple_value=lr)

							train_writer.add_summary(summaries, i)
							train_writer.add_summary(summary, i)

							avg_train_acc[i] += (avg_acc_i / FLAGS.xvalidation_folds)
							avg_train_l2[i] += (avg_l2_loss_i / FLAGS.xvalidation_folds)
							avg_train_entropy[i] += (avg_xentropy_loss_i / FLAGS.xvalidation_folds)
							avg_train_total[i] += (avg_loss_i / FLAGS.xvalidation_folds)

						if i % FLAGS.eval_freq == 0 or i == FLAGS.max_epochs-1:
							# ------------ VALIDATON -------------
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

							avg_val_acc[i] += (tot_acc / FLAGS.xvalidation_folds)
							avg_val_l2[i] += (tot_loss_l2 / FLAGS.xvalidation_folds)
							avg_val_entropy[i] += (tot_loss_entropy / FLAGS.xvalidation_folds)
							avg_val_total[i] += (tot_loss / FLAGS.xvalidation_folds)

							if tot_acc > max_acc:
								max_acc = tot_acc
							print('Validation loss-acc at step %s: %s - %s' % (i, tot_loss, tot_acc))

					# if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
					# checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
					# saver.save(sess, checkpoint_path, global_step=i)

					train_writer.close()
					test_writer.close()
					print('Max validation accuracy in fold %s: %s' % (f, max_acc))

				# Show final kernels from 1st layer
				#				alphas_t, kernels_t = get_kernels(1)
				#				alphas, kernels = sess.run([alphas_t, kernels_t])
				#				show_kernels(kernels)

					dataset.next_fold()

				for i in range(len(avg_val_acc)):
					summary = tf.Summary()
					summary.value.add(tag="Accuracy", simple_value=avg_val_acc[i])
					summary.value.add(tag="Total_Loss", simple_value=avg_val_total[i])
					summary.value.add(tag="X-entropy_Loss", simple_value=avg_val_entropy[i])
					summary.value.add(tag="L2_Loss", simple_value=avg_val_l2[i])

					avg_test_writer.add_summary(summary, i)

					summary = tf.Summary()
					summary.value.add(tag="Accuracy", simple_value=avg_train_acc[i])
					summary.value.add(tag="Total_Loss", simple_value=avg_train_total[i])
					summary.value.add(tag="X-entropy_Loss", simple_value=avg_train_entropy[i])
					summary.value.add(tag="L2_Loss", simple_value=avg_train_l2[i])
					summary.value.add(tag="Learning_rate", simple_value=FLAGS.learning_rate)

					avg_train_writer.add_summary(summary, i)



				avg_train_writer.close()
				avg_test_writer.close()
			except Exception as e:
				# Report exceptions to the coordinator.
				coord.request_stop(e)
			finally:
				# Terminate threads
				coord.request_stop()
				coord.join()
			sess.run(close_queue)
			# Save final model
#			checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
#			saver.save(sess, checkpoint_path)
