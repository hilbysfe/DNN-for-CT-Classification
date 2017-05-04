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

from Models.ae import Autoencoder

LOG_DIR = './logs/AE/'


def train():
	# Set the random seeds for reproducibility. DO NOT CHANGE.
	tf.set_random_seed(42)
	
	batch_size = 128

	# Load data
	# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	# n_samples = mnist.train.num_examples
	cifar10_dataset = cifar10_utils.get_cifar10('./cifar10/cifar-10-batches-py')
	n_samples = cifar10_dataset.train.num_examples

	network_architecture = \
		{
			'Conv_kernels':[11],
			'Conv_maps':[64]
		}  	

	ae = Autoencoder(network_architecture, 
								 learning_rate=0.001)
	
	training_epochs=50
	display_step=5
		
	train_writer = tf.summary.FileWriter(LOG_DIR + '/train', ae.sess.graph)
	
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		# avg_kl = 0.
		# avg_recon = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			# batch_xs = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
			batch_xs, _ = cifar10_dataset.train.next_batch(batch_size)
				
			# Fit training using batch data
			cost = ae.partial_fit(batch_xs)
			
			# Compute average loss
			avg_cost += cost / n_samples * batch_size
			# avg_kl += kl / n_samples * batch_size
			# avg_recon += recon / n_samples * batch_size
			
		# Display logs per epoch step
		if epoch % display_step == 0:
			# Fit training using batch data
			merged = ae.logs(batch_xs)
			summary = tf.Summary(value=[
							tf.Summary.Value(tag="Total_loss", simple_value=avg_cost)
							# tf.Summary.Value(tag="KL_divergence", simple_value=avg_kl),
							# tf.Summary.Value(tag="Reconstruction_loss", simple_value=avg_recon)
						])
			# Add summary
			train_writer.add_summary(summary, epoch)
			train_writer.add_summary(merged, epoch)
			
			print("Epoch:", '%04d' % (epoch+1), 
			  "cost=", "{:.9f}".format(avg_cost))
			
	train_writer.close()	
	
	# -------- Show some reconstructions -----------
	# x_sample = mnist.test.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	x_sample = cifar10_dataset.test.next_batch(batch_size)[0]
	x_reconstruct = ae.reconstruct(x_sample)

	plt.figure(figsize=(8, 12))
	for i in range(5):
		plt.subplot(5, 2, 2*i + 1)
		plt.imshow(x_sample[i,:,:,0])
		plt.title("Test input")
		plt.colorbar()
		plt.subplot(5, 2, 2*i + 2)
		plt.imshow(x_reconstruct[i,:,:,0])
		plt.title("Reconstruction")
		plt.colorbar()
	plt.tight_layout()	
	plt.show()
	
	# -------- Show kernels -----------
	gs = gridspec.GridSpec(8,8)
	gs.update(wspace=0.05, hspace=0.05)
	kernels = ae.get_kernels()
	plt.figure(figsize=(8,8))
	for i in range(8):
		for j in range(8):
			plt.subplot(gs[8*i + j])
			plt.imshow(kernels[8*i + j,:,:,:])
			plt.axis('off')
	plt.show()
	
	# -------- Show TSNE -----------
	labels_test = np.zeros(batch_size*10)
	features = np.zeros((10*batch_size,16384))
	for j in range(10):
		xs, labels = cifar10_dataset.test.next_batch(batch_size)
		# xs, labels = mnist.test.next_batch(batch_size)
		# xs = xs.reshape([batch_size,28,28,1])
				
		i=0
		for label in labels:
			labels_test[i+j*batch_size] = np.where(label==1)[0]
			i=i+1	
		
		f = ae.sess.run(ae.z, feed_dict={ae.x: xs})
		
		features[j*batch_size:(j+1)*batch_size,:] = f.reshape((batch_size, np.shape(f)[1]*np.shape(f)[2]*np.shape(f)[3]))
		
		
	model = TSNE(n_components = 2, random_state=0)
	Y = model.fit_transform(np.array(features))
	
	y1=Y[labels_test==1]
	y2=Y[labels_test==2]
	y3=Y[labels_test==3]
	y4=Y[labels_test==4]
	y5=Y[labels_test==5]
	y6=Y[labels_test==6]
	y7=Y[labels_test==7]
	y8=Y[labels_test==8]
	y9=Y[labels_test==9]
	y0=Y[labels_test==0]

	# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	colors = ['blue', 'red', 'green', 'yellow', 'cyan', 'purple', 'darkorange', 'lime', 'lightcoral', 'magenta', ]

	ax0 = plt.scatter(y0[:,0], y0[:,1], 30, colors[0]);
	ax1 = plt.scatter(y1[:,0], y1[:,1], 30, colors[1]);
	ax2 = plt.scatter(y2[:,0], y2[:,1], 30, colors[2]);
	ax3 = plt.scatter(y3[:,0], y3[:,1], 30, colors[3]);
	ax4 = plt.scatter(y4[:,0], y4[:,1], 30, colors[4]);
	ax5 = plt.scatter(y5[:,0], y5[:,1], 30, colors[5]);
	ax6 = plt.scatter(y6[:,0], y6[:,1], 30, colors[6]);
	ax7 = plt.scatter(y7[:,0], y7[:,1], 30, colors[7]);
	ax8 = plt.scatter(y8[:,0], y8[:,1], 30, colors[8]);
	ax9 = plt.scatter(y9[:,0], y9[:,1], 30, colors[9]);


	plt.legend((ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9),
				classes,
				scatterpoints=1,
				loc='upper right',
				ncol=1,
				fontsize=9)
	plt.show();
	
	# -------- Train SVM on hidden representation -----------
	trs = 1000
	tes = 1000
	# X_train, labels_train = mnist.train.next_batch(trs)
	# X_val, labels_val = mnist.test.next_batch(tes)
	
	X_train, labels_train = cifar10_dataset.train.next_batch(trs)
	X_val, labels_val = cifar10_dataset.test.next_batch(tes)

	X_train = X_train.reshape(trs, -1)	
	X_val = X_val.reshape(tes, -1)
	
	labels_train = [label.tolist().index(1) for label in labels_train]
	labels_val = [label.tolist().index(1) for label in labels_val]
	
	classifier = OneVsRestClassifier(
					svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0))).fit(X_train, labels_train)
	
	Y_val = classifier.predict(X_val)
	
	acc = np.sum(Y_val==labels_val)/tes
	print('Multi-class Accuracy of raw data: %s' %(acc))
	
	X_train, labels_train = cifar10_dataset.train.next_batch(trs)
	X_train = ae.sess.run(ae.z, feed_dict={ae.x: X_train})
	X_val, labels_val = cifar10_dataset.test.next_batch(tes)
	X_val = ae.sess.run(ae.z, feed_dict={ae.x: X_val})
	
	X_train = X_train.reshape(trs, -1)	
	X_val = X_val.reshape(tes, -1)
	
	labels_train = [label.tolist().index(1) for label in labels_train]
	labels_val = [label.tolist().index(1) for label in labels_val]
	
	classifier = OneVsRestClassifier(
					svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0))).fit(X_train, labels_train)
	
	Y_val = classifier.predict(X_val)
	acc = np.sum(Y_val==labels_val)/tes	
	print('Multi-class Accuracy of learnt representation: %s' %(acc))
	
	
def main(_):
	
	if not tf.gfile.Exists(LOG_DIR):
		tf.gfile.MakeDirs(LOG_DIR)
	else:
		shutil.rmtree(LOG_DIR)
		tf.gfile.MakeDirs(LOG_DIR)
	
	train()
	
if __name__ == '__main__':
	
	tf.app.run()
