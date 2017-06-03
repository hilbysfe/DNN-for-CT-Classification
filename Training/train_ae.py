from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import input_data
import cifar10_utils
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
import sklearn.svm as svm
import argparse

from Models.ae import Autoencoder

LOG_DIR = './logs/AE/'
LEARNING_RATE = 0.001
BATCH_SIZE = 128
CHECKPOINT_PATH = './ae_models/model.ckpt'

def _loss(x, x_reconstr, fmaps, sp_lambda):
	# --- Reconstruction loss ---
	reconstr_loss = tf.reduce_mean(tf.square(x_reconstr - x))
	
	# --- Sparsity loss ---
	# L1_sparsity = 0.0
	# for map in fmaps:
		# L1_sparsity += tf.divide(tf.norm(map, ord=1, axis=(1, 2)), len(fmaps)) # impose uniform feature maps
	# sparsity_loss = tf.reduce_mean(L1_sparsity)
	
	loss = reconstr_loss #+ sp_lambda * sparsity_loss
	
	return loss#reconstr_loss, sparsity_loss, loss

def train_step(loss, global_step):
	# decay_steps = 390*50
	# LEARNING_RATE_DECAY_FACTOR = 0.5

	# Decay the learning rate exponentially based on the number of steps.
	# lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                  # global_step,
                                  # decay_steps,
                                  # LEARNING_RATE_DECAY_FACTOR,
                                  # staircase=True)
	# tf.summary.scalar('learning_rate', lr)
	train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
	return train_op

def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	global_step = tf.contrib.framework.get_or_create_global_step()
	
	# Load data
	cifar10_dataset = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
	n_samples = cifar10_dataset.train.num_examples

	def feed_dict(train):
		if train:
			xs, ys = cifar10_dataset.train.next_batch(BATCH_SIZE)
		else:
			xs, ys = cifar10_dataset.test.next_batch(BATCH_SIZE)
		return {x: xs, is_training: train}
	
	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
		is_training = tf.placeholder(tf.bool, name='is-training')
	
	display_step=5
		
	network_architecture = \
	{
		'Conv_kernels':[11],
		'Conv_maps':[64],
		'Conv_strides':[[1,2,2,1]],
		'Basis':[10]
	}  	

	ae = {
			'RFNN' 			: lambda: Autoencoder(network_architecture, rfnn=True, sigmas=FLAGS.sigmas),
			'CTNET'			: lambda: Autoencoder(network_architecture, rfnn=False)
		}[FLAGS.model_name]()
	
	reconstr, layer1 = ae.inference(x)
	# reconstr_loss, sparsity_loss, loss = _loss(x, reconstr, [ae.z], ae.sp_lambda)
	loss = _loss(x, reconstr, [0], ae.sp_lambda)
	
	# Call optimizer
	train_op = train_step(loss, global_step)
	
	# Define session
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		
		train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	
		# Training cycle
		for epoch in range(FLAGS.max_epochs):
			avg_cost = 0.
			avg_rl = 0.
			avg_sp = 0.
			total_batch = int(n_samples / BATCH_SIZE)
			# Loop over all batches
			for i in range(total_batch):
				# _, rloss, sloss, total_loss = sess.run([train_op, reconstr_loss, sparsity_loss, loss], feed_dict=feed_dict(True))
				_, total_loss = sess.run([train_op, loss], feed_dict=feed_dict(True))
				
				# Compute average loss
				avg_cost += total_loss / n_samples * BATCH_SIZE
				# avg_rl += rloss / n_samples * BATCH_SIZE
				# avg_sp += sloss / n_samples * BATCH_SIZE
				
			# Display logs per epoch step
			if epoch % display_step == 0:
				# Fit training using batch data
				# merged = ae.logs(batch_xs)
				summary = tf.Summary(value=[
								tf.Summary.Value(tag="Total_loss", simple_value=avg_cost)
								# tf.Summary.Value(tag="KL_divergence", simple_value=avg_kl),
								# tf.Summary.Value(tag="Reconstruction_loss", simple_value=avg_recon)
							])
				# Add summary
				train_writer.add_summary(summary, epoch)
				# train_writer.add_summary(merged, epoch)
				
				print("Epoch:", '%04d' % (epoch+1), 
				  "cost=", "{:.9f}".format(avg_cost))
				  # "recon=", "{:.9f}".format(avg_rl),
				  # "sparse=", "{:.9f}".format(avg_sp))
				
		train_writer.close()
		if FLAGS.model_name == 'RFNN':
			np.save(FLAGS.pretrained_weights_path, sess.run(ae.alphas))
		else:
			np.save(FLAGS.pretrained_weights_path, sess.run(ae.kernels))
		np.save(FLAGS.pretrained_biases_path, sess.run(ae.biases))
	
		# -------- Show some reconstructions -----------
		# x_sample = mnist.test.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
		# x_sample = cifar10_dataset.test.next_batch(BATCH_SIZE)[0]		
		x_reconstruct, layer = sess.run([reconstr,layer1],  feed_dict=feed_dict(False))
		
		plt.figure(figsize=(8, 12))
		for i in range(5):
			plt.subplot(5, 1, 2*i + 1)
			plt.imshow(x_reconstruct[i,:,:])
			plt.title("Reconstruction")
		plt.tight_layout()	
		plt.show()
		
		gs = gridspec.GridSpec(8,8)
		gs.update(wspace=0.05, hspace=0.05)
		plt.figure(figsize=(8,8))
		for i in range(8):
			for j in range(8):
				plt.subplot(gs[8*i + j])
				plt.imshow(layer[0,:,:,8*i + j])
				plt.axis('off')
		plt.show()
		
		# gs = gridspec.GridSpec(8,8)
		# gs.update(wspace=0.05, hspace=0.05)
		# plt.figure(figsize=(8,8))
		# for i in range(8):
			# for j in range(8):
				# plt.subplot(gs[8*i + j])
				# plt.imshow(latent[0,:,:,8*i + j])
				# plt.axis('off')
		# plt.show()
		
		# -------- Show kernels -----------
		gs = gridspec.GridSpec(8,8)
		gs.update(wspace=0.05, hspace=0.05)
		kernels = sess.run(ae.get_kernels())
		plt.figure(figsize=(8,8))
		for i in range(8):
			for j in range(8):
				plt.subplot(gs[8*i + j])
				plt.imshow(kernels[8*i + j,:,:])
				plt.axis('off')
		plt.show()
	
	# # -------- Show TSNE -----------
	# labels_test = np.zeros(batch_size*10)
	# features = np.zeros((10*batch_size,16384))
	# for j in range(10):
		# xs, labels = cifar10_dataset.test.next_batch(batch_size)
		# # xs, labels = mnist.test.next_batch(batch_size)
		# # xs = xs.reshape([batch_size,28,28,1])
				
		# i=0
		# for label in labels:
			# labels_test[i+j*batch_size] = np.where(label==1)[0]
			# i=i+1	
		
		# f = ae.sess.run(ae.z, feed_dict={ae.x: xs})
		
		# features[j*batch_size:(j+1)*batch_size,:] = f.reshape((batch_size, np.shape(f)[1]*np.shape(f)[2]*np.shape(f)[3]))

		
	# model = TSNE(n_components = 2, random_state=0)
	# Y = model.fit_transform(np.array(features))
	
	# y1=Y[labels_test==1]
	# y2=Y[labels_test==2]
	# y3=Y[labels_test==3]
	# y4=Y[labels_test==4]
	# y5=Y[labels_test==5]
	# y6=Y[labels_test==6]
	# y7=Y[labels_test==7]
	# y8=Y[labels_test==8]
	# y9=Y[labels_test==9]
	# y0=Y[labels_test==0]

	# # classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
	# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	# colors = ['blue', 'red', 'green', 'yellow', 'cyan', 'purple', 'darkorange', 'lime', 'lightcoral', 'magenta', ]

	# ax0 = plt.scatter(y0[:,0], y0[:,1], 30, colors[0]);
	# ax1 = plt.scatter(y1[:,0], y1[:,1], 30, colors[1]);
	# ax2 = plt.scatter(y2[:,0], y2[:,1], 30, colors[2]);
	# ax3 = plt.scatter(y3[:,0], y3[:,1], 30, colors[3]);
	# ax4 = plt.scatter(y4[:,0], y4[:,1], 30, colors[4]);
	# ax5 = plt.scatter(y5[:,0], y5[:,1], 30, colors[5]);
	# ax6 = plt.scatter(y6[:,0], y6[:,1], 30, colors[6]);
	# ax7 = plt.scatter(y7[:,0], y7[:,1], 30, colors[7]);
	# ax8 = plt.scatter(y8[:,0], y8[:,1], 30, colors[8]);
	# ax9 = plt.scatter(y9[:,0], y9[:,1], 30, colors[9]);


	# plt.legend((ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9),
				# classes,
				# scatterpoints=1,
				# loc='upper right',
				# ncol=1,
				# fontsize=9)
	# plt.show();
	
	# # -------- Train SVM on hidden representation -----------
	# trs = 1000
	# tes = 1000
	# # X_train, labels_train = mnist.train.next_batch(trs)
	# # X_val, labels_val = mnist.test.next_batch(tes)
	
	# X_train, labels_train = cifar10_dataset.train.next_batch(trs)
	# X_val, labels_val = cifar10_dataset.test.next_batch(tes)

	# X_train = X_train.reshape(trs, -1)	
	# X_val = X_val.reshape(tes, -1)
	
	# labels_train = [label.tolist().index(1) for label in labels_train]
	# labels_val = [label.tolist().index(1) for label in labels_val]
	
	# classifier = OneVsRestClassifier(
					# svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0))).fit(X_train, labels_train)
	
	# Y_val = classifier.predict(X_val)
	
	# acc = np.sum(Y_val==labels_val)/tes
	# print('Multi-class Accuracy of raw data: %s' %(acc))
	
	# X_train, labels_train = cifar10_dataset.train.next_batch(trs)
	# X_train = ae.sess.run(ae.z, feed_dict={ae.x: X_train})
	# X_val, labels_val = cifar10_dataset.test.next_batch(tes)
	# X_val = ae.sess.run(ae.z, feed_dict={ae.x: X_val})
	
	# X_train = X_train.reshape(trs, -1)	
	# X_val = X_val.reshape(tes, -1)
	
	# labels_train = [label.tolist().index(1) for label in labels_train]
	# labels_val = [label.tolist().index(1) for label in labels_val]
	
	# classifier = OneVsRestClassifier(
					# svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0))).fit(X_train, labels_train)
	
	# Y_val = classifier.predict(X_val)
	# acc = np.sum(Y_val==labels_val)/tes	
	# print('Multi-class Accuracy of learnt representation: %s' %(acc))
	
	
	return ae
	
def main(_):
	
	if not tf.gfile.Exists(LOG_DIR):
		tf.gfile.MakeDirs(LOG_DIR)
	else:
		shutil.rmtree(LOG_DIR)
		tf.gfile.MakeDirs(LOG_DIR)
	
	train()
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--pretrained_weights_path', type = str, default = "./pretrained_weights/weights.npy",
						help='Specify the path to write the weights to.')	
	parser.add_argument('--pretrained_biases_path', type = str, default = "./pretrained_weights/biases.npy",
						help='Specify the path to write the weights to.')	
	parser.add_argument('--max_epochs', type = int, default = 100,
						help='Iterations to train the AE for.')						
	parser.add_argument('--model_name', type = str, default = 'CTNET',
						help='Model name')	
	parser.add_argument('--sigmas', type = str,
						help='Sigmas for RFNN')						
	FLAGS, unparsed = parser.parse_known_args()
	
	if 'RFNN' in FLAGS.model_name:
		FLAGS.sigmas = [float(x) for x in FLAGS.sigmas.split(',')]
	
	tf.app.run()
