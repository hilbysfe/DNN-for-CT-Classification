
from cifar import cifar10_utils

from Utils import utils
from Models.rfnn import RFNN
from Models.ctnet import CTNET
from Models.densenet import DenseNet
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

def train_cifar(FLAGS, NUM_GPUS):
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	with tf.Graph().as_default():
		with tf.device('/cpu:0'):

			global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

			# ====== LOAD DATASET ======
			print('Loading Dataset...')
			#            cifar10_dataset = cifar10_utils.get_cifar10('/home/nicolab/Downloads/cifar-10-batches-py')
			cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.cifar_path)
			print('Loading Dataset...done.')

			# ====== DEFINE SPACEHOLDERS ======
			with tf.name_scope('input'):
				image_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, 32, 32, 3], name='x-input')
				label_batch = tf.placeholder(tf.float32, [NUM_GPUS, FLAGS.batch_size, 10], name='y-input')
				is_training = tf.placeholder(tf.bool, name='is-training')
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
						xi, yi = cifar10_dataset.test.next_batch(FLAGS.batch_size)
						xs.append(xi)
						ys.append(yi)
				elif flag == 2:
					for i in np.arange(NUM_GPUS):
						xi = cifar10_dataset.train.images[0:FLAGS.batch_size]
						yi = cifar10_dataset.train.labels[0:FLAGS.batch_size]
						xs.append(xi)
						ys.append(yi)
				return {image_batch: xs, label_batch: ys, is_training: flag == 0 or flag == 2}

			# ====== MODEL DEFINITION ======
			print('Defining model...')

			sigmas = [float(x) for x in FLAGS.sigmas.split(',')]
			kernels = [int(x) for x in FLAGS.kernels.split(',')]
			maps = [int(x) for x in FLAGS.maps.split(',')]
			strides = [int(x) for x in FLAGS.strides.split(',')]

			#            model = RFNN(
			#                n_classes=10,
			#                kernels=kernels,
			#                maps=maps,
			#                sigmas=sigmas,
			#                bases=bases,
			#                bases_3d=FLAGS.bases3d,
			#                is_training=is_training,
			#                batchnorm=FLAGS.batch_normalization
			#            )
			#            model = CTNET(
			#                n_classes=10,
			#                kernels=kernels,
			#                maps=maps,
			#                strides=strides,
			#                pretraining=False,
			#                is_training = is_training,
			#                conv3d=False,
			#                bnorm=FLAGS.batch_normalization)

			# model = DenseNet(
			# 	growth_rate=FLAGS.growth_rate,
			# 	depth=FLAGS.depth,
			# 	total_blocks=FLAGS.total_blocks,
			# 	keep_prob=FLAGS.keep_prob,
			# 	model_type=FLAGS.model_type,
			# 	is_training=is_training,
			# 	init_kernel=FLAGS.init_kernel,
			# 	comp_kernel=FLAGS.comp_kernel,
			# 	reduction=FLAGS.reduction,
			# 	bc_mode=FLAGS.bc_mode,
			# 	n_classes=10
			# )

			model = RFNNDenseNet(
				growth_rate=FLAGS.growth_rate,
				depth=FLAGS.depth,
				total_blocks=FLAGS.total_blocks,
				keep_prob=FLAGS.keep_prob,
				model_type=FLAGS.model_type,
				is_training=is_training,
				init_kernel=FLAGS.init_kernel,
				comp_kernel=FLAGS.comp_kernel,
				sigmas=sigmas,
				init_order=FLAGS.init_order,
				comp_order=FLAGS.comp_order,
				reduction=FLAGS.reduction,
				bc_mode=FLAGS.bc_mode,
				n_classes=10
			)
			print('Defining model...done.')

			# ====== DEFINE LOSS, ACCURACY TENSORS ======
			print('Defining necessary OPs...')

			#            opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
			opt = tf.train.MomentumOptimizer(
				FLAGS.learning_rate, FLAGS.nesterov_momentum, use_nesterov=True)

			# === DEFINE QUEUE OPS ===
			batch_queue = tf.FIFOQueue(
				capacity=NUM_GPUS,
				dtypes=[tf.float32, tf.float32],
				shapes=[(FLAGS.batch_size, 32, 32, 3), (FLAGS.batch_size, 10)]
			)
			batch_enqueue = batch_queue.enqueue_many([image_batch, label_batch])
			close_queue = batch_queue.close()

		# Calculate the gradients for each model tower.
		tower_grads = []
		tower_losses = []
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

						loss = tower_loss_dense(logits, y, FLAGS.weight_decay, scope)
						accuracy = tower_accuracy(logits, y, scope)
						#                        exp_signal = exp_GB(signal, FLAGS.alpha)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						# Calculate the gradients for the batch of data on this tower.
						grads = opt.compute_gradients(loss)

						# Keep track of the gradients across all towers.
						tower_grads.append(grads)
						tower_losses.append(loss)
						tower_accuracies.append(accuracy)
		with tf.device('/cpu:0'):

			# Calculate the mean of each gradient - synchronization point across towers.
			avg_grads = average_gradients(tower_grads)
			avg_loss = tf.reduce_mean(tower_losses, 0)
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

			# Print initial kernels
			# alphas_tensor, kernels_tensor = get_kernels()
			# alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])
			# np.save('./Kernels/kernel_0.npy', kernels_array)
			# np.save('./Kernels/alphas_0.npy', alphas)

			# ====== UPDATE VARIABLES ======

			print('Defining update OPs...')

			# Apply the gradients to adjust the shared variables.
			train_op = opt.apply_gradients(avg_grads, global_step=global_step)

			# Track the moving averages of all trainable variables.
			#            variable_averages = tf.train.ExponentialMovingAverage(
			#                        0.9999, global_step)
			#            variables_averages_op = variable_averages.apply(tf.trainable_variables())

			# Group all updates into a single train op.
			#            train_op = tf.group(apply_gradient_op, variables_averages_op)

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
			for f in range(1):
				try:
					training_steps = int(cifar10_dataset.train.num_examples / (NUM_GPUS * FLAGS.batch_size))
					validation_steps = int(cifar10_dataset.test.num_examples / (NUM_GPUS *FLAGS.batch_size))
					sess.run(tf.global_variables_initializer())

					lr = FLAGS.learning_rate

					#                    print([n.name for n in tf.get_default_graph().as_graph_def().node
					#                               if 'ConvLayer1' in n.name and 'alphas' in n.name])

					# Show initial kernels from 1st layer
					#                    alphas_t, kernels_t = get_kernels(1)
					#                    alphas, kernels = sess.run([alphas_t, kernels_t])
					#                    show_kernels(kernels)

					# Create a coordinator, launch the queue runner threads.
					coord = tf.train.Coordinator()
					tf.train.start_queue_runners(sess, coord=coord)

					# Init writers
					train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train/' + str(f), sess.graph)
					test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test/' + str(f))

					# ======== LSUV INIT WEIGHTS WITH A FORWARD PASS ==========
					print("Initializing weights with LSUV...")
					# Conv layers
					for l in range(len(model.alphas)):
						var = 0.0
						t_i = 0
						while abs(var - 1.0) >= FLAGS.tol_var and t_i < FLAGS.t_max:
							sess.run(batch_enqueue, feed_dict=feed_dict(0))
							alphas, b_l = sess.run([model.alphas[l], model.conv_act[l]], feed_dict={is_training: False})
							var = np.var(b_l)
#							print(var)
							sess.run(model.alphas[l].assign(alphas/np.sqrt(var)))
							t_i += 1
					# Bottleneck layers
					print("bottleneck")
					for l in range(len(model.bc_conv_act)):
						var = 0.0
						t_i = 0
						while abs(var - 1.0) >= FLAGS.tol_var and t_i < FLAGS.t_max:
							sess.run(batch_enqueue, feed_dict=feed_dict(0))
							w_l, b_l = sess.run([model.bc_weights[l], model.bc_conv_act[l]], feed_dict={is_training: False})
							var = np.var(b_l)
#							print(var)
							sess.run(model.bc_weights[l].assign(w_l / np.sqrt(var)))
							t_i += 1
					print("Initializing weights with LSUV...done.")

					max_acc = 0
					for i in range(int(FLAGS.max_epochs * training_steps)):

						if coord.should_stop():
							break

						# ----- Reduce learning rate ------
						if i == FLAGS.reduce_lr_epoch_1 * training_steps:
							lr = lr / 10
						if i == FLAGS.reduce_lr_epoch_2 * training_steps:
							lr = lr / 10

						# ------------ TRAIN -------------
						if i % (FLAGS.print_freq * training_steps) == 0:
							# ------------ PRINT -------------
							sess.run(batch_enqueue, feed_dict=feed_dict(0))
							_, summaries, loss_value, acc_value = \
								sess.run([train_op, summary_op, avg_loss, avg_accuracy],
										 feed_dict={is_training: True, learning_rate: lr})

							summary = tf.Summary()
							summary.value.add(tag="Accuracy", simple_value=acc_value)
							summary.value.add(tag="Loss", simple_value=loss_value)
							summary.value.add(tag="Learning_rate", simple_value=lr)

							train_writer.add_summary(summaries, i)
							train_writer.add_summary(summary, i)
						else:
							sess.run(batch_enqueue, feed_dict=feed_dict(0))
							_, loss_value = sess.run([train_op, avg_loss],
													 feed_dict={is_training: True, learning_rate: lr})

						assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

						# ------------ VALIDATON -------------
						if i % (FLAGS.eval_freq * training_steps) == 0 or i == int(FLAGS.max_epochs * training_steps):
							tot_acc = 0.0
							tot_loss = 0.0
							for step in range(validation_steps):
								sess.run(batch_enqueue, feed_dict=feed_dict(1))
								acc_s, loss_s = sess.run([avg_accuracy, avg_loss], feed_dict={is_training: False})
								tot_acc += (acc_s / validation_steps)
								tot_loss += (loss_s / validation_steps)

							#                                print(sm)
							#                                print(y_)

							summary = tf.Summary()
							summary.value.add(tag="Accuracy", simple_value=tot_acc)
							summary.value.add(tag="Loss", simple_value=tot_loss)

							test_writer.add_summary(summary, i)

							#						if tot_acc > max_acc:
							#							max_acc = tot_acc
							#						print('Validation accuracy at step %s: %s' % (i, tot_acc))
							print('Validation loss at step %s: %s' % (i, tot_loss))

					# if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
					# checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
					# saver.save(sess, checkpoint_path, global_step=i)

					train_writer.close()
					test_writer.close()
					print('Max validation accuracy in fold %s: %s' % (f, max_acc))

				# Show final kernels from 1st layer
				#                    alphas_t, kernels_t = get_kernels(1)
				#                    alphas, kernels = sess.run([alphas_t, kernels_t])
				#                    show_kernels(kernels)

				except Exception as e:
					# Report exceptions to the coordinator.
					coord.request_stop(e)
				finally:
					# Terminate threads
					coord.request_stop()
					coord.join()
			sess.run(close_queue)

# Print final kernels
# alphas_tensor, kernels_tensor = get_kernels()
# alphas, kernels_array = sess.run([alphas_tensor, kernels_tensor])
# np.save('./Kernels/kernel_final.npy', kernels_array)
# np.save('./Kernels/alphas_final.npy', alphas)

