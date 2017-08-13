import numpy as np
import tensorflow as tf

from Utils.cnn_utils import _conv_layer_2d
from Utils.cnn_utils import _deconv_layer_2d
from Utils.cnn_utils import _deconv_layer_tied

from Utils.rfnn_utils import init_basis_hermite_2D
from Utils.rfnn_utils import init_basis_hermite_3D
from Utils.rfnn_utils import _rfnn_conv_layer_2d
from Utils.rfnn_utils import _rfnn_deconv_layer_2d

			
def floatX(X):
	return np.asarray(X, dtype=np.float32)

def corrupt(x):
    return tf.multiply(x, tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=0.2,
                                               dtype=tf.float32))
	
def keep_top_k(x, k):	
	values, indices = tf.nn.top_k(-x, k)
	
	a = np.zeros((x.get_shape()[0].value, x.get_shape()[1].value, x.get_shape()[2].value, len(x.get_shape().as_list())-1))
	for i in range(x.get_shape()[0].value):
		for j in range(x.get_shape()[1].value):
			for h in range(x.get_shape()[2].value):
				a[i,j,h,:] = [i,j,h]
	a = tf.convert_to_tensor(a, dtype=tf.int32)
	
	full_indices = tf.concat(3, [a, tf.expand_dims(indices[:,:,:,0], 3)])
	for i in range(1, k):
		full_indices = tf.concat(3, [full_indices, 
									 tf.concat(3, [a, tf.expand_dims(indices[:,:,:,i], 3)])]
								)
	full_indices2 = tf.reshape(full_indices, [-1, len(x.get_shape().as_list())])
	
	to_substract = tf.sparse_to_dense(full_indices2, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
	
	return x + to_substract
	
def spatial_sparsity(x):
	return keep_top_k(x, 1)
	
def lifetime_sparsity(x, lifetime):
	return tf.transpose(keep_top_k(tf.transpose(x, [3, 1, 2, 0]), lifetime), [3, 0, 1, 2])	
	
def _WTA_layer(input, lifetime='None'):
	net = spatial_sparsity(input)
	
	if lifetime != 'None':
		net = lifetime_sparsity(net, lifetime)
	
	return net
	
def _structured_sparsity_layer(x):
	# first across featuremap
	examp_sparsity = tf.norm(x, ord=2, axis=3, keep_dims=True)
	net = tf.divide(x, examp_sparsity+1e-9)
	
	# then per featuremap
	net_vec = tf.reshape(net, [-1, net.get_shape()[3].value, net.get_shape()[1].value*net.get_shape()[2].value])
	feat_sparsity = tf.norm(net_vec, ord=2, axis=2, keep_dims=False)
	feat_sparsity = tf.expand_dims(feat_sparsity, 1)
	feat_sparsity = tf.expand_dims(feat_sparsity, 2)
	net = tf.divide(net, feat_sparsity+1e-9)
	
	return net

	
	
class Autoencoder(object):
	
	def __init__(self, network_architecture, rfnn=False, sigmas=[], sp_lambda=1):
		self.network_architecture = network_architecture
		self.initializer = tf.contrib.layers.xavier_initializer()
		self.sp_lambda = sp_lambda
		self.rfnn = rfnn
		
		if self.rfnn:
			self.sigmas = sigmas	
			self.hermit	= init_basis_hermite_2D					
			self.basis = [ self.hermit(network_architecture['Conv_kernels'][i], sigmas, network_architecture['Basis'][i])
								for i in range(len(network_architecture['Conv_kernels'])) ]			

	def load_weights(self, x, path_weights, path_biases, is_training):
		weights = np.load(path_weights)
		biases = np.load(path_biases)
		assign_ops = []
		with tf.variable_scope('AutoEncoder'):
			with tf.variable_scope('Encoder'):
				net = x
				for i, (kernel, map, strides) in enumerate(zip(self.network_architecture['Conv_kernels'],
										   self.network_architecture['Conv_maps'],
										   self.network_architecture['Conv_strides'])):
					with tf.variable_scope('ConvLayer'+str(i+1)):
						if self.rfnn:
							alphas, bias, net, k = _rfnn_conv_layer_2d(
									input=net,
									basis=self.basis[i],
									omaps=map, 
									strides=strides, 
									padding='SAME',
									is_training=is_training,
									bnorm=False)
							assign_ops.append(alphas.assign(weights[i]))
							assign_ops.append(bias.assign(biases[i]))
						else:
							net, k, bias = _conv_layer_2d(
									input=net,
									shape=[kernel, kernel, net.get_shape()[3], map], 
									strides=strides, 
									padding='SAME', 
									is_training=is_training,
									bnorm=False)
							assign_ops.append(k.assign(weights[i]))
							assign_ops.append(bias.assign(biases[i]))
						
		return assign_ops, net

	def inference(self, x):
		with tf.variable_scope('AutoEncoder'):
			print(x.get_shape())
			encoder = []
			self.alphas = []
			self.kernels = []
			self.biases = []
			with tf.variable_scope('Encoder'):
				# ==== Encoder ====
				# net = corrupt(x)
				net = x
				for i, (kernel, map, strides) in enumerate(zip(self.network_architecture['Conv_kernels'],
															   self.network_architecture['Conv_maps'],
															   self.network_architecture['Conv_strides'])):
					with tf.variable_scope('ConvLayer'+str(i+1)):
						encoder.append(net)
						
						if self.rfnn:
							alphas, bias, net, k = _rfnn_conv_layer_2d(
									input=net,
									basis=self.basis[i],
									omaps=map, 
									strides=strides, 
									padding='SAME',
									is_training=True,
									bnorm=False)
							net = tf.stack(net)
							net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
							
							self.alphas.append(alphas)
							
						else:
							net, k, bias = _conv_layer_2d(
									input=net, 
									shape=[kernel, kernel, net.get_shape()[3], map], 
									strides=strides, 
									padding='SAME',
									is_training=True,
									bnorm=False)
									
							self.kernels.append(k)
						self.biases.append(bias)
						print(net.get_shape())		
			self.layer1 = net
			# with tf.variable_scope('Structured_sparsity'):
				# self.z = _structured_sparsity_layer(net)
				# net = self.z
				# print(net.get_shape())
			
			with tf.variable_scope('Decoder'):
				for i, (kernel, strides) in enumerate(zip(reversed(self.network_architecture['Conv_kernels']),
														  reversed(self.network_architecture['Conv_strides']))):
					shape = tf.shape(encoder[-1-i])
					oshape = tf.stack([shape[0], shape[1], shape[2], shape[3]])
					with tf.variable_scope('DeconvLayer'+str(i+1)):
						if self.rfnn:
							net = _rfnn_deconv_layer_2d(
									input=net,
									basis=self.basis[i],
									omaps=encoder[-1-i].get_shape()[-1].value,
									oshape=oshape,
									strides=strides, 
									padding='SAME',
									bnorm=False)
							# check which rotation was chosen in Encoder part?
							net = tf.stack(net)
							net = tf.stack(tf.reduce_max(net, reduction_indices=[0]))
						else:
							net = _deconv_layer_2d(
									input=net, 
									kshape=[kernel, kernel, encoder[-(i+1)].get_shape()[-1].value, net.get_shape()[3].value], 
									oshape=oshape,
									strides=strides, 
									padding='SAME')
								
						
						print(net.get_shape())
			
			_activation_summary(net)
			return net, self.layer1
			
		
	def reconstruct(self, X):
		return self.inference(X)
		
		
							 
	def logs(self, X):
		merged = tf.summary.merge_all()
		
		return self.sess.run(merged, feed_dict={self.x: X})
		
	
	def get_kernels(self):			
		kernel = tf.get_default_graph().get_tensor_by_name("AutoEncoder/Encoder/ConvLayer_0/weights:0")
		kernel_avg = tf.reduce_mean(kernel, axis=2)
		kernel_transposed = tf.transpose(kernel_avg, [2, 0, 1])
			
		return kernel_transposed

def _activation_summary(x):
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name, x)
							 
