from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import input_data
import cifar10_utils
import shutil
import matplotlib.pyplot as plt

from Models.vae import VariationalAutoencoder

LOG_DIR = './logs/VAE/'

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
		dict(n_hidden_recog_1=500, 	# 1st layer encoder neurons
			 n_hidden_recog_2=500, 	# 2nd layer encoder neurons
			 n_hidden_gener_1=500, 	# 1st layer decoder neurons
			 n_hidden_gener_2=500, 	# 2nd layer decoder neurons
			 n_input=3072, 			# img shape: 28*28
			 n_z=60)  				# dimensionality of latent space

	vae = VariationalAutoencoder(network_architecture, 
								 learning_rate=0.001, 
								 batch_size=batch_size)
	
	training_epochs=100
	display_step=20
		
	train_writer = tf.summary.FileWriter(LOG_DIR + '/train', vae.sess.graph)
	
	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		avg_kl = 0.
		avg_recon = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			# batch_xs, _ = mnist.train.next_batch(batch_size)
			batch_xs, _ = cifar10_dataset.train.next_batch(batch_size)
				
			# Fit training using batch data
			cost, kl, recon = vae.partial_fit(batch_xs.reshape((batch_size, 3072)))
			
			# Compute average loss
			avg_cost += cost / n_samples * batch_size
			avg_kl += kl / n_samples * batch_size
			avg_recon += recon / n_samples * batch_size
			
		# Display logs per epoch step
		if epoch % display_step == 0:
			# Fit training using batch data
			merged = vae.logs(batch_xs.reshape((batch_size, 3072)))
			summary = tf.Summary(value=[
							tf.Summary.Value(tag="Total_loss", simple_value=avg_cost),
							tf.Summary.Value(tag="KL_divergence", simple_value=avg_kl),
							tf.Summary.Value(tag="Reconstruction_loss", simple_value=avg_recon)
						])
			# Add summary
			train_writer.add_summary(summary, epoch)
			train_writer.add_summary(merged, epoch)
			
			print("Epoch:", '%04d' % (epoch+1), 
			  "cost=", "{:.9f}".format(avg_cost))
			
	
	
	
	# x_sample = mnist.test.next_batch(batch_size)[0]
	x_sample = cifar10_dataset.test.next_batch(batch_size)[0]
	x_reconstruct = vae.reconstruct(x_sample.reshape((batch_size, 3072)))

	plt.figure(figsize=(8, 12))
	for i in range(5):

		plt.subplot(5, 2, 2*i + 1)
		plt.imshow(x_sample[i].reshape(32, 32, 3), vmin=0, vmax=1)
		plt.title("Test input")
		plt.colorbar()
		plt.subplot(5, 2, 2*i + 2)
		plt.imshow(x_reconstruct[i].reshape(32, 32, 3), vmin=0, vmax=1)
		plt.title("Reconstruction")
		plt.colorbar()
	plt.tight_layout()
	
	plt.show()
	
	train_writer.close()
	
def main(_):
	
	if not tf.gfile.Exists(LOG_DIR):
		tf.gfile.MakeDirs(LOG_DIR)
	else:
		shutil.rmtree(LOG_DIR)
		tf.gfile.MakeDirs(LOG_DIR)
	
	train()
	
if __name__ == '__main__':
	
	tf.app.run()
