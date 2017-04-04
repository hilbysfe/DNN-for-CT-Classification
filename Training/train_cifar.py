from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from convnet import ConvNet
from siamese import Siamese

import cifar10_utils
import cifar10_siamese_utils


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
	"""
	Defines the ops to conduct an optimization step. You can set a learning
	rate scheduler or pick your favorite optimizer here. This set of operations
	should be applicable to both ConvNet() and Siamese() objects.

	Args:
		loss: scalar float Tensor, full loss = cross_entropy + reg_loss

	Returns:
		train_op: Ops for optimization.
	"""
	########################
	#	PUT YOUR CODE HERE	#
	########################
	
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
	
	########################
	# END OF YOUR CODE		#
	########################

	return train_op

	

def train():
	"""
	Performs training and evaluation of ConvNet model.

	First define your graph using class ConvNet and its methods. Then define
	necessary operations such as trainer (train_step in this case), savers
	and summarizers. Finally, initialize your model within a tf.Session and
	do the training.

	---------------------------
	How to evaluate your model:
	---------------------------
	Evaluation on test set should be conducted over full batch, i.e. 10k images,
	while it is alright to do it over minibatch for train set.

	---------------------------------
	How often to evaluate your model:
	---------------------------------
	- on training set every print_freq iterations
	- on test set every eval_freq iterations

	------------------------
	Additional requirements:
	------------------------
	Also you are supposed to take snapshots of your model state (i.e. graph,
	weights and etc.) every checkpoint_freq iterations. For this, you should
	study TensorFlow's tf.train.Saver class. For more information, please
	checkout:
	[https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
	"""

	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	np.random.seed(42)

	########################
	# PUT YOUR CODE HERE	#
	########################
		
	def feed_dict(train):
		if train:
			xs, ys = cifar10_dataset.train.next_batch(FLAGS.batch_size)
		else:
			xs, ys = cifar10_dataset.test.images, cifar10_dataset.test.labels
		return {x: xs, y_: ys, is_training: train}	
	
	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
	
	# Define session
	sess = tf.InteractiveSession()

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		is_training = tf.placeholder(tf.bool, name='is-training')
		
	# Define and initialize network	
	net = ConvNet(10, is_training)
	
	# Calculate predictions
	logits = net.inference(x)	
	loss = net.loss(logits, y_)
	accuracy = net.accuracy(logits, y_)
	# confusion = net.confusion(logits, y_)
		
	# Call optimizer
	train_op = train_step(loss)
	
	# Create a saver.
	saver = tf.train.Saver(tf.all_variables())
	
	# Merge all the summaries and write them out to log
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',	sess.graph)
	test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
	tf.initialize_all_variables().run()
	
	# Train
	max_acc = 0
	for i in range(1, 1+FLAGS.max_steps):
		# ------------ TRAIN -------------
		_ = sess.run([train_op], feed_dict=feed_dict(True))
		if i % FLAGS.eval_freq == 0 or i == 1:				
			# ------------ TEST -------------
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			if acc > max_acc:
				#checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
				#saver.save(sess, checkpoint_path)
				max_acc = acc
			print('Accuracy at step %s: %s' % (i, acc))
					
#			if i == FLAGS.max_steps:
#				np.savetxt("confusion_matrix.csv", conf.astype(int), delimiter=",")
						
		
#			print('Adding run metadata for', i)
		elif i % FLAGS.print_freq == 0:
			# ------------ PRINT -------------
			summary = sess.run([merged], feed_dict=feed_dict(True))
			train_writer.add_summary(summary[0], i)
		
		if i % FLAGS.checkpoint_freq == 0: # or i == FLAGS.max_steps:
			checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=i)
	train_writer.close()
	test_writer.close()
	print('Max accuracy : %s' % (max_acc))

	########################
	# END OF YOUR CODE		#
	########################



def train_siamese():
	"""
	Performs training and evaluation of Siamese model.

	First define your graph using class Siamese and its methods. Then define
	necessary operations such as trainer (train_step in this case), savers
	and summarizers. Finally, initialize your model within a tf.Session and
	do the training.

	---------------------------
	How to evaluate your model:
	---------------------------
	On train set, it is fine to monitor loss over minibatches. On the other
	hand, in order to evaluate on test set you will need to create a fixed
	validation set using the data sampling function you implement for siamese
	architecture. What you need to do is to iterate over all minibatches in
	the validation set and calculate the average loss over all minibatches.

	---------------------------------
	How often to evaluate your model:
	---------------------------------
	- on training set every print_freq iterations
	- on test set every eval_freq iterations

	------------------------
	Additional requirements:
	------------------------
	Also you are supposed to take snapshots of your model state (i.e. graph,
	weights and etc.) every checkpoint_freq iterations. For this, you should
	study TensorFlow's tf.train.Saver class. For more information, please
	checkout:
	[https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
	"""

	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	np.random.seed(42)

	########################
	# PUT YOUR CODE HERE	#
	########################
	def feed_dict_train():
		_left, _right, ys = cifar10_siamese_dataset.train.next_batch(
				FLAGS.batch_size)
		train = True
		_margin = float(0.2)
		
		return {left: _left, right: _right, y_: ys,
						is_training: train, margin: _margin}

	# Load data
	cifar10_siamese_dataset = cifar10_siamese_utils.get_cifar10(
			FLAGS.data_dir, validation_size=1000, one_hot=False)
	val_dset = cifar10_siamese_utils.create_dataset(
			cifar10_siamese_dataset.validation)
	test_dset = cifar10_siamese_utils.create_dataset(
			cifar10_siamese_dataset.test)

	# Define session
	# sess = tf.InteractiveSession()

	siamese_model = Siamese()

	with tf.Graph().as_default():

		# Input placeholders
		with tf.name_scope('input'):
			left = tf.placeholder(
					tf.float32,
					[None, 32, 32, 3],
					name='left-input')
			right = tf.placeholder(
					tf.float32,
					[None, 32, 32, 3],
					name='right-input')
			y_ = tf.placeholder(
					tf.float32,
					[None],
					name='y-input')
			is_training = tf.placeholder(
					tf.bool,
					name='training-input')
			margin = tf.placeholder(
					tf.float32,
					name='margin-input')

		with tf.name_scope("inference"):
			# with tf.name_scope("Channel_1"):
			channel_1 = siamese_model.inference(left, False)
			# with tf.name_scope("Channel_2"):
			channel_2 = siamese_model.inference(right, True)

		with tf.name_scope('contrastive_loss'):
			loss = siamese_model.loss(channel_1, channel_2, y_, margin)

		# Call optimizer
		with tf.name_scope('train_op'):
			train_op = train_step(loss)

		merged = tf.merge_all_summaries()

		init = tf.initialize_all_variables()
		sess = tf.Session()

		# summary writers
		summary_writer = tf.train.SummaryWriter(
				FLAGS.log_dir +
				'/train',
				sess.graph)
		summary_writer_val = tf.train.SummaryWriter(
				FLAGS.log_dir +
				'/val',
				sess.graph)

		sess.run(init)

		saver = tf.train.Saver()

		best_test_loss, best_val_loss = 1e3, 1e3
		for i in range(FLAGS.max_steps):

			_ = sess.run([train_op], feed_dict=feed_dict_train())
			if i % int(FLAGS.print_freq) == 0:

				# if not val_dset:
				val_loss = 0
				for _left, _right, ys in val_dset:

					feed_dict_val = {
							left: _left,
							right: _right,
							y_: ys,
							is_training: False,
							margin: float(0.2)}

					val_summary, batch_loss = sess.run(
							[merged, loss], feed_dict=feed_dict_val)
					val_loss += batch_loss

				val_loss /= len(val_dset)

				# summ_train, loss_train = sess.run(
				#		 [summary, loss], feed_dict=feed_dict_val)
				summary_writer.add_summary(val_summary, i)

				print(
						'{0:40s} {1:10d}'.format(
								'validation loss is: {0:.3g} @iteration:'.format(
										float(val_loss)),
								int(i)))

			if (i % int(FLAGS.eval_freq) == 0 and i != 0) or i == int(
							FLAGS.max_steps)-1:

				##########################
				#	 Perform validation	 #
				##########################

				# if not val_dset:
				test_loss = 0
				for _left, _right, ys in test_dset:

					feed_dict_test = {
							left: _left,
							right: _right,
							y_: ys,
							is_training: False,
							margin: float(0.2)}

					test_summary, batch_loss = sess.run(
							[merged, loss], feed_dict=feed_dict_test)
					test_loss += batch_loss

				test_loss /= len(test_dset)

				summary_writer_val.add_summary(test_summary, i)

				print(
						'{0:40s} {1:10d}'.format(
								'average test loss: {0:.4g} @iteration:'.format(test_loss),
								int(i)))

			# write checkpoints
			if(i % int(FLAGS.checkpoint_freq) == 0) or (i == int(FLAGS.max_steps)-1):

				print(
						'{0:40s} {1:10d}'.format(
								'saving @iteration: {0:21s}'.format(' '),
								int(i)))
			# if i % FLAGS.checkpoint_freq == 0:	# or i == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=i)

#		print(
#			 'Best val loss: {0:.3g}, Best test loss: {1:.3g}'.format(
#						best_train_loss,
#						best_test_loss))
	########################
	# END OF YOUR CODE		#
	########################


def feature_extraction():
	"""
	This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
	model with restored parameters. From then on you can basically use that model in
	any way you want, for instance, feature extraction, finetuning or as a submodule
	of a larger architecture. However, this method should extract features from a
	specified layer and store them in data files such as '.h5', '.npy'/'.npz'
	depending on your preference. You will use those files later in the assignment.

	Args:
		[optional]
	Returns:
		None
	"""

	########################
	# PUT YOUR CODE HERE	#
	########################
	
	# ==================== CNN ======================
	
	# Load data
#	cifar10_dataset = cifar10_utils.get_cifar10(FLAGS.data_dir)
	
#	xs_train, ys_train = cifar10_dataset.train.next_batch(20000)
	#xs_test, ys_test = cifar10_dataset.test.images, cifar10_dataset.test.labels
	
	# Define session
#	sess = tf.InteractiveSession()
	
	# Input placeholders
#	with tf.name_scope('input'):
#		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
#		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
#		is_training = tf.placeholder(tf.bool, name='is-training')
		
	# Define and initialize network	
#	net = ConvNet(10)
	
	# Calculate predictions
#	logits = net.inference(x)	
#	loss = net.loss(logits, y_)
#	accuracy = net.accuracy(logits, y_)		
	
	# Get feature activation
#	flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/Flatten/Activation:0")
#	fc1 = tf.get_default_graph().get_tensor_by_name("ConvNet/FullLayer1/Activation:0")
#	fc2 = tf.get_default_graph().get_tensor_by_name("ConvNet/FullLayer2/Activation:0")

	# Create a saver.
#	saver = tf.train.Saver(tf.all_variables())
	
	# Init variables
#	tf.initialize_all_variables().run()	
	
	# Restore variables from checkpoint
#	checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'best_model.ckpt')
#	saver.restore(sess, checkpoint_path)	
#	print("Model loaded.")	
	
	# Run inference
#	merged = tf.merge_all_summaries()	
	
#	_, acc, flatten_np_train, fc1_np_train, fc2_np_train = sess.run([merged, accuracy, flatten, fc1, fc2], {x:xs_train, y_:ys_train})
#	_, acc, flatten_np_test, fc1_np_test, fc2_np_test = sess.run([merged, accuracy, flatten, fc1, fc2], {x:xs_test, y_:ys_test})		
	
#	labels_train = np.zeros(ys_train.shape[0])
#	i=0
#	for label in ys_train:
#		labels_train[i] = np.where(label==1)[0]
#		i=i+1
	
#	labels_test = np.zeros(ys_test.shape[0])
#	i=0
#	for label in ys_test:
#		labels_test[i] = np.where(label==1)[0]
#		i=i+1	
	
	# Save features and labels
#	np.save("./features/flatten_train.npy", flatten_np_train)
#	np.save("./features/fc1_train.npy", fc1_np_train)
#	np.save("./features/fc2_train.npy", fc2_np_train)
#	np.save("./features/labels_train.npy", labels_train)

#	np.save("./features/flatten_test.npy", flatten_np_test)
#	np.save("./features/fc1_test.npy", fc1_np_test)
#	np.save("./features/fc2_test.npy", fc2_np_test)
#	np.save("./features/labels_test.npy", labels_test)
	
	# =============== Siamese =================
	
	print("Siamese feature_extraction.")	
	
	# Load data
	cifar10_siamese_dataset = cifar10_siamese_utils.get_cifar10(
			FLAGS.data_dir, validation_size=1000, one_hot=False)

	test_dataset = cifar10_siamese_utils.create_dataset(cifar10_siamese_dataset.test, num_tuples=1, batch_size=2000)

	# Define session
	sess = tf.InteractiveSession()

	siamese_model = Siamese()

	# Input placeholders
	with tf.name_scope('input'):
		left = tf.placeholder(
				tf.float32,
				[None, 32, 32, 3],
				name='left-input')
		right = tf.placeholder(
				tf.float32,
				[None, 32, 32, 3],
				name='right-input')
		y_ = tf.placeholder(
				tf.float32,
				[None],
				name='y-input')
		is_training = tf.placeholder(
				tf.bool,
				name='training-input')
		margin = tf.placeholder(
				tf.float32,
				name='margin-input')

	with tf.name_scope("inference"):
		# with tf.name_scope("Channel_1"):
		channel_1 = siamese_model.inference(left, False)
		# with tf.name_scope("Channel_2"):
		channel_2 = siamese_model.inference(right, True)

	with tf.name_scope('contrastive_loss'):
		loss = siamese_model.loss(channel_1, channel_2, y_, margin)

#	for n in tf.get_default_graph().as_graph_def().node:
#		if "L2_Normalization" in n.name:
#		print(n.name) 

	l2norm = tf.get_default_graph().get_tensor_by_name("inference/ConvNet/FullyConnected/L2_Normalization/Div:0")
	
	print("l2norm found.")
	
	# Create a saver.
	saver = tf.train.Saver(tf.all_variables())
		
	# Init variables
	tf.initialize_all_variables().run()	
	
	# Restore variables from checkpoint
	checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
	saver.restore(sess, checkpoint_path)	
	print("Model loaded.")	
	

	feed_dict_test = {
		left: test_dataset[0][0],
		right: test_dataset[0][1],
		y_: test_dataset[0][2],
		is_training: False,
		margin: float(0.2)}
			
	_, l2norm_np = sess.run([loss, l2norm], feed_dict=feed_dict_test)

	
	np.save("./features/l2norm_2000.npy", l2norm_np)
	np.save("./features/labels_2000.npy", test_dataset[0][2])
	
	########################
	# END OF YOUR CODE		#
	########################

def initialize_folders():
	"""
	Initializes all folders in FLAGS variable.
	"""

	if not tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.MakeDirs(FLAGS.log_dir)

	if not tf.gfile.Exists(FLAGS.data_dir):
		tf.gfile.MakeDirs(FLAGS.data_dir)

	if not tf.gfile.Exists(FLAGS.checkpoint_dir):
		tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
	"""
	Prints all entries in FLAGS variable.
	"""
	for key, value in vars(FLAGS).items():
		print(key + ' : ' + str(value))

def main(_):
	print_flags()

	initialize_folders()
	if FLAGS.is_train == True:
		if FLAGS.train_model == 'linear':
			train()
		elif FLAGS.train_model == 'siamese':
			train_siamese()
		else:
			raise ValueError("--train_model argument can be linear or siamese")
	else:
		feature_extraction()

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
						help='Learning rate')
	parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
						help='Number of steps to run trainer.')
	parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
						help='Batch size to run trainer.')
	parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
						help='Frequency of evaluation on the train set')
	parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
						help='Frequency of evaluation on the test set')
	parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
						help='Frequency with which the model state is saved.')
	parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
						help='Directory for storing input data')
	parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
						help='Summaries log directory')
	parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
						help='Checkpoint directory')
	parser.add_argument('--is_train', type = str, default = True,
						help='Training or feature extraction')
	parser.add_argument('--train_model', type = str, default = 'linear',
						help='Type of model. Possible options: linear and siamese')

	FLAGS, unparsed = parser.parse_known_args()

	tf.app.run()
