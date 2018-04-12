import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_experiment_infos(FLAGS):
	l = [
		"learning_rate: " + str(FLAGS.learning_rate),
		"max_epochs: " + str(FLAGS.max_epochs),
		"batch_size: " + str(FLAGS.batch_size),
		"pretraining: " + str(FLAGS.pretraining),
		"xvalidation_folds: " + str(FLAGS.xvalidation_folds),
		"normalization: " + str(FLAGS.normalization),
		"batch_normalization: " + str(FLAGS.batch_normalization),
		"sigmas: " + str(FLAGS.sigmas),
		"kernels: " + str(FLAGS.kernels),
		"maps: " + str(FLAGS.maps),
		"bases: " + str(FLAGS.bases),
		"bases3d: " + str(FLAGS.bases3d),
		"print_freq: " + str(FLAGS.print_freq),
		"eval_freq: " + str(FLAGS.eval_freq),
		"log_dir: " + str(FLAGS.log_dir),
		"trainingpath: " + str(FLAGS.trainingpath),
		"testpath: " + str(FLAGS.testpath),
	]
	return tf.convert_to_tensor(l)

def get_kernels(i):
	kernel = tf.get_default_graph().get_tensor_by_name("tower_0/ConvLayer%d/weights_0:0" % i)
	alphas = tf.get_default_graph().get_tensor_by_name("ConvLayer%d/alphas:0" % i)
	#    print(kernel.get_shape())

	kernel_avg = tf.reduce_mean(kernel, axis=2)
	#    print(kernel_avg.get_shape())
	x_min = tf.reduce_min(kernel_avg)
	x_max = tf.reduce_max(kernel_avg)
	kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)

	# to tf.image_summary format [batch_size, height, width, channels]
	kernel_transposed = tf.transpose(kernel_avg, [2, 0, 1])
	# print(kernel_transposed.get_shape())

	return alphas, kernel_transposed

def show_kernels(kernels):
	k = int(np.sqrt(np.shape(kernels)[-1]))
	f, axarr = plt.subplots(k, k)
	f.set_figheight(2*k)
	f.set_figwidth(2*k)
	f.subplots_adjust(hspace=0.05, wspace=0.05)
	for i in range(k):
		for j in range(k):
			axarr[j, i].imshow(kernels[:,:,i * k + j], cmap='gray')
			axarr[j, i].set_axis_off()

def exp_GB(logits, alpha):
	return tf.multiply(tf.exp(logits), alpha)

def pow3_GB(logits, alpha, beta):
	return tf.add(tf.multiply(tf.pow(logits, 3), alpha), beta)

def tower_accuracy(logits, labels):
	softmax = tf.nn.softmax(logits)
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
	#        tf.summary.scalar('Accuracy', accuracy)

	return accuracy, correct_prediction, softmax

def tower_accuracy_exp(logits, labels, scope):
	#    softmax = tf.nn.softmax(logits)
	#    signal = exp_GB(logits, FLAGS.alpha)
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	with tf.name_scope('Accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#        tf.summary.scalar('Accuracy', accuracy)

	return accuracy, correct_prediction, logits

def tower_loss(logits, labels, scope):
	with tf.name_scope('Cross_Entropy_Loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels,
																name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)
	with tf.name_scope('Total_Loss'):
		total_loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')
	#        tf.summary.scalar('Total_loss', total_loss)
	return total_loss

def tower_loss_dense(logits, labels):
	with tf.name_scope('Cross_Entropy_Loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels,
																name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

	with tf.name_scope('L2_Loss'):
		l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')

	return cross_entropy_mean, l2_loss

def tower_loss_exp(logits, labels, alpha, scope):
	with tf.name_scope('Cross_Entropy_Loss'):
		#        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
		#        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		signal = exp_GB(logits, alpha)
		sqdiff = tf.squared_difference(signal, labels)
		mean_rms = tf.reduce_mean(sqdiff)
		tf.add_to_collection('losses', mean_rms)
	with tf.name_scope('Total_Loss'):
		total_loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')
	#        tf.summary.scalar('Total_loss', total_loss)
	return total_loss, sqdiff

def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, v in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)

	return average_grads
