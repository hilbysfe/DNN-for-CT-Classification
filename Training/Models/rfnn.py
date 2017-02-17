from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import convolve1d as conv

def batch_norm_wrapper(inputs, is_training, is_conv, decay = 0.999):
	
	epsilon = 1e-3
	
	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	def train_case():
		if is_conv:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2,3])
		else:
			batch_mean, batch_var = tf.nn.moments(inputs,[0])
			
		train_mean = tf.assign(pop_mean,
							   pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
							  pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, epsilon)
	def test_case():
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, epsilon)
			
	return tf.cond(is_training, train_case, test_case)
		

def floatX(X):
	return np.asarray(X, dtype=np.float32)

def init_basis_hermite_2D(sigma,bases):
	filterExtent = 3*sigma
	x = np.arange(-filterExtent, filterExtent+1, dtype=np.float)
	imSize = filterExtent*2+1
	impulse = np.zeros( (np.int(imSize), np.int(imSize)) )
	
	impulse[np.int(imSize/2), np.int(imSize/2)] = 1.0
	nrBasis = 15
	hermiteBasis = np.empty( (np.int(nrBasis), np.int(imSize), np.int(imSize)) )
	g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
	g = g/g.sum()
	g1 = sigma * -(x/ np.square(sigma)) * g
	g2 = np.square(sigma) * ( (np.square(x)-np.power(sigma,2)) / np.power(sigma,4)) * g
	g3 = np.power(sigma,3) * -( (np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
	g4 = np.power(sigma,4) * ( ( (np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8) ) ) * g
	gauss0x = conv(impulse, g, axis=1)
	gauss0y = conv(impulse, g, axis=0)
	gauss1x = conv(impulse, g1, axis=1)
	gauss1y = conv(impulse, g1, axis=0)
	gauss2x = conv(impulse, g2, axis=1)
	gauss0 = conv(gauss0x, g, axis=0)
	hermiteBasis[0,:,:] = gauss0		# g
	vmax = gauss0.max()
	vmin = -vmax
	#print vmax, vmin
	hermiteBasis[1,:,:] = conv(gauss0y, g1, axis=1) # g_x
	hermiteBasis[2,:,:] = conv(gauss0x, g1, axis=0) # g_y
	
	hermiteBasis[3,:,:] = conv(gauss0y, g2, axis=1) # g_xx
	hermiteBasis[4,:,:] = conv(gauss0x, g2, axis=0) # g_yy
	hermiteBasis[5,:,:] = conv(gauss1x, g1, axis=0) # g_xy
	
	hermiteBasis[6,:,:] = conv(gauss0y, g3, axis=1) # g_xxx
	hermiteBasis[7,:,:] = conv(gauss0x, g3, axis=0) # g_yyy
	hermiteBasis[8,:,:] = conv(gauss1y, g2, axis=1) # g_xxy
	hermiteBasis[9,:,:] = conv(gauss1x, g2, axis=0) # g_yyx
	
	hermiteBasis[10,:,:] = conv(gauss0y, g4, axis=1) # g_xxxx
	hermiteBasis[11,:,:] = conv(gauss0x, g4, axis=0) # g_yyyy
	hermiteBasis[12,:,:] = conv(gauss1y, g3, axis=1) # g_xxxy
	hermiteBasis[13,:,:] = conv(gauss1x, g3, axis=0) # g_yyyx
	hermiteBasis[14,:,:] = conv(gauss2x, g2, axis=0) # g_yyxx
	
	return tf.constant(floatX(hermiteBasis[0:bases,:,:]))

def init_basis_hermite_3D(sigma,bases):
	filterExtent = 3*sigma
	x = np.arange(-filterExtent, filterExtent+1, dtype=np.float)
	imSize = filterExtent*2+1
	impulse = np.zeros( (np.int(imSize), np.int(imSize), np.int(imSize)) )
	impulse[(np.int(imSize))/2,(np.int(imSize))/2, np.int(imSize)/2] = 1.0
	nrBasis = 35
	hermiteBasis = np.empty( (np.int(nrBasis), np.int(imSize), np.int(imSize), np.int(imSize)) )
	g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
	g = g/g.sum()
	g1 = sigma * -(x/ np.square(sigma)) * g
	g2 = np.square(sigma) * ( (np.square(x)-np.power(sigma,2)) / np.power(sigma,4)) * g
	g3 = np.power(sigma,3) * -( (np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
	g4 = np.power(sigma,4) * ( ( (np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8) ) ) * g
	
	# 0. order: g
	gauss0 = conv(conv(conv(impulse, g, axis=0), g, axis=1), g, axis=2)

	# 1. order: g_z, g_x, g_y
	gauss1z_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g1, axis=2)
	gauss1x_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g1, axis=1)
	gauss1y_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g1, axis=0)

	# 2. order: g_xx, g_yy, g_zz, 
	#		   g_xy, g_xz, g_zy
	gauss2xx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g2, axis=1)
	gauss2yy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g2, axis=0)
	gauss2zz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g2, axis=2)

	gauss2xy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g1, axis=0)
	gauss2xz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g1, axis=2)
	gauss2zy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g1, axis=0)

	# 3. order: g_xxx, g_yyy, g_zzz
	#		   g_xxy, g_xxz, g_xyy, g_xzz, g_yyz, g_yzz, g_xyz
	gauss3xxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g3, axis=1)
	gauss3yyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g3, axis=0)
	gauss3zzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g3, axis=2)

	gauss3xxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g2, axis=1)
	gauss3xxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g2, axis=1)
	gauss3xyy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g2, axis=1)
	gauss3xzz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g2, axis=2)
	gauss3yyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g2, axis=0)
	gauss3yzz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g2, axis=2)
	gauss3xyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g1, axis=2)

	# 4. order: g_xxxx, g_yyyy, g_zzzz, 
	#		   g_xxxy, g_xxxz, g_yyyx, g_yyyz, g_zzzx, g_zzzy, 
	#		   g_xxyy, g_xxzz, g_xxyz, g_xyyz, g_yyzz, g_xzzy,
	gauss4xxxx_ = conv(conv(conv(impulse, g, axis=0), g, axis=2), g4, axis=1)
	gauss4yyyy_ = conv(conv(conv(impulse, g, axis=1), g, axis=2), g4, axis=0)
	gauss4zzzz_ = conv(conv(conv(impulse, g, axis=0), g, axis=1), g4, axis=2)

	gauss4xxxy_ = conv(conv(conv(impulse, g, axis=2), g1, axis=0), g3, axis=1)
	gauss4xxxz_ = conv(conv(conv(impulse, g, axis=0), g1, axis=2), g3, axis=1)
	gauss4yyyx_ = conv(conv(conv(impulse, g, axis=2), g1, axis=1), g3, axis=0)
	gauss4yyyz_ = conv(conv(conv(impulse, g, axis=1), g1, axis=2), g3, axis=0)
	gauss4zzzx_ = conv(conv(conv(impulse, g, axis=0), g1, axis=1), g3, axis=2)
	gauss4zzzy_ = conv(conv(conv(impulse, g, axis=1), g1, axis=0), g3, axis=2)

	gauss4xxyy_ = conv(conv(conv(impulse, g, axis=2), g2, axis=0), g2, axis=1)
	gauss4xxzz_ = conv(conv(conv(impulse, g, axis=0), g2, axis=2), g2, axis=1)
	gauss4xxyz_ = conv(conv(conv(impulse, g1, axis=0), g1, axis=2), g2, axis=1)
	gauss4xyyz_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=2), g2, axis=0)
	gauss4yyzz_ = conv(conv(conv(impulse, g, axis=1), g2, axis=0), g2, axis=2)
	gauss4xzzy_ = conv(conv(conv(impulse, g1, axis=1), g1, axis=0), g2, axis=2)
	
	hermiteBasis[0,:,:,:] = gauss0
	hermiteBasis[1,:,:,:] = gauss1z_
	hermiteBasis[2,:,:,:] = gauss1x_
	hermiteBasis[3,:,:,:] = gauss1y_
	hermiteBasis[4,:,:,:] = gauss2xx_
	hermiteBasis[5,:,:,:] = gauss2yy_
	hermiteBasis[6,:,:,:] = gauss2zz_
	hermiteBasis[7,:,:,:] = gauss2xy_
	hermiteBasis[8,:,:,:] = gauss2xz_
	hermiteBasis[9,:,:,:] = gauss2zy_
	hermiteBasis[10,:,:,:] = gauss3xxx_
	hermiteBasis[11,:,:,:] = gauss3yyy_
	hermiteBasis[12,:,:,:] = gauss3zzz_
	hermiteBasis[13,:,:,:] = gauss3xxy_
	hermiteBasis[14,:,:,:] = gauss3xxz_
	hermiteBasis[15,:,:,:] = gauss3xyy_
	hermiteBasis[16,:,:,:] = gauss3xzz_
	hermiteBasis[17,:,:,:] = gauss3yyz_
	hermiteBasis[18,:,:,:] = gauss3yzz_
	hermiteBasis[19,:,:,:] = gauss3xyz_
	hermiteBasis[20,:,:,:] = gauss4xxxx_
	hermiteBasis[21,:,:,:] = gauss4yyyy_
	hermiteBasis[22,:,:,:] = gauss4zzzz_
	hermiteBasis[23,:,:,:] = gauss4xxxy_
	hermiteBasis[24,:,:,:] = gauss4xxxz_
	hermiteBasis[25,:,:,:] = gauss4yyyx_
	hermiteBasis[26,:,:,:] = gauss4yyyz_
	hermiteBasis[27,:,:,:] = gauss4zzzx_
	hermiteBasis[28,:,:,:] = gauss4zzzy_
	hermiteBasis[29,:,:,:] = gauss4xxyy_
	hermiteBasis[30,:,:,:] = gauss4xxzz_
	hermiteBasis[31,:,:,:] = gauss4xxyz_
	hermiteBasis[32,:,:,:] = gauss4xyyz_
	hermiteBasis[33,:,:,:] = gauss4yyzz_
	hermiteBasis[34,:,:,:] = gauss4xzzy_
	
	return tf.constant(floatX(hermiteBasis[0:bases,:,:,:]))
	
def init_bias(units, name):
	return tf.get_variable(
		name,
		shape=units,
		initializer=tf.constant_initializer(0.0))

def init_alphas(nrFilters,channels,nrBasis,name):
	return tf.get_variable(
		name,
		shape=[nrFilters,channels,nrBasis],
		initializer=tf.random_uniform_initializer(-1.0, 1.0))
	
def init_weights(shape, name):
	return tf.get_variable(
		name,
		shape=shape,
		initializer=tf.random_normal_initializer(0.01))
	




class RFNN(object):

	def __init__(self, n_classes = 10, bases_3d = False, sigmas=[0.5, 1, 1.5], dropout_rate_conv=.2, dropout_rate_hidden=0.7, is_training = True):
	
		self.n_classes				= n_classes
		self.dropout_rate_conv		= dropout_rate_conv
		self.dropout_rate_hidden	= dropout_rate_hidden
		self.is_training			= is_training
		
		self.normalizer				= tf.nn.lrn
		self.act					= tf.nn.relu
		
		self.hermit					= init_basis_hermite_3D if bases_3d else init_basis_hermite_2D
		
		self.inference				= self.inference_3d if bases_3d else self.inference_2d
		#-------------------------
		# Init Basis and Alphas
		#-------------------------

		bases_L1 = 10
		sigma_L1 = sigmas[0]
		# bases_L2 = 15
		# sigma_L2 = sigmas[1]
		# bases_L3 = 15
		# sigma_L3 = sigmas[2]
		
		# bases_L4 = 10
		# sigma_L4 = sigmas[3]
		# bases_L5 = 10
		# sigma_L5 = sigmas[4]
		# bases_L6 = 10
		# sigma_L6 = sigmas[5]
		
		# bases_L7 = 10
		# sigma_L7 = sigmas[6]
		
		self.basis_L1 = self.hermit(sigma_L1,bases_L1)
		# self.basis_L2 = self.hermit(sigma_L2,bases_L2)
		# self.basis_L3 = self.hermit(sigma_L3,bases_L3)		
		# self.basis_L4 = self.hermit(sigma_L4,bases_L4)
		# self.basis_L5 = self.hermit(sigma_L5,bases_L5)
		# self.basis_L6 = self.hermit(sigma_L6,bases_L6)
		# self.basis_L7 = self.hermit(sigma_L7,bases_L7)

		self.alphas_L1 = init_alphas(64,1,bases_L1, name="L1_alphas")
		# self.alphas_L2 = init_alphas(160,192,bases_L2, name="L2_alphas")
		# self.alphas_L3 = init_alphas(96,160,bases_L3, name="L3_alphas")
		
		# self.alphas_L4 = init_alphas(192,96,bases_L4, name="L4_alphas")
		# self.alphas_L5 = init_alphas(160,192,bases_L5, name="L5_alphas")
		# self.alphas_L6 = init_alphas(96,160,bases_L6, name="L6_alphas")
		
		# self.alphas_L7 = init_alphas(64,96,bases_L7, name="L7_alphas")

		
		
	def _conv_layer_2d(self, input, kernel, stride, padding, name, bnorm=True):
		with tf.variable_scope(name) as scope:						
			net = tf.nn.conv2d(input, kernel, strides=stride, padding=padding)
			if bnorm:
				net = batch_norm_wrapper(net, self.is_training, True)
			net = self.act(net, name=scope.name)			
			self._activation_summary(net)
			
		with tf.variable_scope(name + '/visualization'):
			# scale weights to [0 1], type is still float
			kernel_avg = tf.reduce_mean(kernel, axis=2)
			x_min = tf.reduce_min(kernel_avg)
			x_max = tf.reduce_max(kernel_avg)
			kernel_0_to_1 = (kernel_avg - x_min) / (x_max - x_min)
						
			# to tf.image_summary format [batch_size, height, width, channels]
			kernel_transposed = tf.transpose(kernel_0_to_1, [2, 0, 1])
			kernel_transposed = tf.expand_dims(kernel_transposed, axis=3)
			batch = kernel_transposed.get_shape()[0].value
						
			tf.summary.image('/filters', kernel_transposed, max_outputs=batch)
			
		return net
		
	def _conv_layer_3d(self, input, kernel, stride, padding, name, bnorm=True):
		with tf.variable_scope(name) as scope:						
			net = tf.nn.conv3d(input, kernel, strides=stride, padding=padding)
			if bnorm:
				net = batch_norm_wrapper(net, self.is_training, True)
			net = self.act(net, name=scope.name)			
			self._activation_summary(net)
			
		with tf.variable_scope(name + '/visualization'):
			# scale weights to [0 1], type is still float
			x_min = tf.reduce_min(kernel)
			x_max = tf.reduce_max(kernel)
			kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)

			# to tf.image_summary format [batch_size, height, width, channels]
			kernel_transposed = tf.transpose (kernel_0_to_1, [4, 0, 1, 2, 3])

			tf.summary.image(name + '/filters', kernel_transposed)
			
		return net
		

	#---------------------------
	# Model Definition
	#---------------------------

	# ---- Use this for 2D models ----
	def inference_2d(self, X):
		print(X.get_shape())
		ch = X.get_shape()[3].value
		
		# --- Define weights ---
		w = tf.reduce_sum(
			tf.transpose(self.alphas_L1[:,:,:,None,None]) *
			tf.transpose(self.basis_L1[None,None,:,:,:])
			,axis = 2)		
		w_L1 = tf.tile(w, [1, 1, ch, 1])
		# w_L2 = tf.reduce_sum(
			# tf.transpose(self.alphas_L2[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L2[None,None,:,:,:])
			# ,axis = 2)
		# w_L3 = tf.reduce_sum(
			# tf.transpose(self.alphas_L3[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L3[None,None,:,:,:])
			# ,axis = 2)
		# w_L4 = tf.reduce_sum(
			# tf.transpose(self.alphas_L4[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L4[None,None,:,:,:])
			# ,axis = 2)
		# w_L5 = tf.reduce_sum(
			# tf.transpose(self.alphas_L5[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L5[None,None,:,:,:])
			# ,axis = 2)
		# w_L6 = tf.reduce_sum(
			# tf.transpose(self.alphas_L6[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L6[None,None,:,:,:])
			# ,axis = 2)
		# w_L7 = tf.reduce_sum(
			# tf.transpose(self.alphas_L7[:,:,:,None,None]) * 
			# tf.transpose(self.basis_L7[None,None,:,:,:])
			# ,axis = 2)	
		
		# ==== Layer 1 ====				
		l1 = self._conv_layer_2d(
				input=X,
				kernel=w_L1,
				stride=[1,1,1,1],
				padding='SAME',
				name='ConvLayer1a')
		# l1 = self._conv_layer_2d(
				# input=l1,
				# kernel=w_L2,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer1b')
		# l1 = self._conv_layer_2d(
				# input=l1,
				# kernel=w_L3,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer1c')				
		# print(l1.get_shape())
		
		l1 = tf.nn.max_pool(l1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID")
		# print(l1.get_shape())
		
		l1 = batch_norm_wrapper(l1, self.is_training, True)
		
		# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
		# l1 = tf.nn.dropout(l1, keep_prob)

		# ==== Layer 2 ====			
		# l2 = self._conv_layer_2d(
				# input=l1,
				# kernel=w_L4,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer2a')
		# l2 = self._conv_layer_2d(
				# input=l2,
				# kernel=w_L5,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer2b')
		# l2 = self._conv_layer_2d(
				# input=l2,
				# kernel=w_L6,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer2c')				
		# print(l2.get_shape())
		
		# l2 = tf.nn.max_pool(l2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID")
		# print(l2.get_shape())
					
		# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
		# l2 = tf.nn.dropout(l2, 1-self.dropout_rate_conv)

		# ==== Layer 3 ====		
		# l3 = self._conv_layer_2d(
				# input=l2,
				# kernel=w_L7,
				# stride=[1,1,1,1],
				# padding='SAME',
				# name='ConvLayer3')
		# print(l3.get_shape())
		
		# l3 = tf.nn.max_pool(l3, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID")
		# print(l3.get_shape())
						
		# ==== Layer 4 ====		
		with tf.variable_scope('Flatten'):
			fshape = l1.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value
			l4 = tf.reshape(l1, [-1, dim])
		print(l4.get_shape())		

		# keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
		# pyx = tf.nn.dropout(l4, 1-self.dropout_rate_hidden)
		
		self.w_L4 = init_weights((dim, self.n_classes), name="L8_weights")
		
		with tf.variable_scope('FullLayer'):
			pyx = tf.matmul(l4, self.w_L4)
			pyx = batch_norm_wrapper(pyx, self.is_training, False)
		
		# with tf.variable_scope('AVG_Pooling'):
			# pyx = tf.nn.avg_pool(l3, ksize=[1,14,14,1], strides=[1,1,1,1], padding="VALID")
			# pyx = batch_norm_wrapper(pyx, self.is_training, False)
		
		# print(pyx.get_shape())		
		
		return pyx
		
	# ---- Use this for 3D models ----
	def inference_3d(self, X):
		print(X.get_shape())
		
		# ==== Layer 1 ====		
		w_L1 = tf.reduce_sum(
			tf.transpose(self.alphas_L1[:,:,:,None,None]) *
			tf.transpose(self.basis_L1[None,None,:,:,:,:])
			,axis = 2)		
		l1b = self._conv_layer_3d(
				input=X,
				kernel=w_L1,
				stride=[1,1,1,1,1],
				padding='SAME',
				name='ConvLayer1')
		print(l1b.get_shape())
		
		l1 = tf.nn.max_pool3d(l1b, ksize=[1,3,3,2,1], strides=[1,2,2,2,1], padding="VALID")
		print(l1.get_shape())
				
#		keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
#		l1 = tf.nn.dropout(l1, keep_prob)

		# ==== Layer 2 ====		
		w_L2 = tf.reduce_sum(
			tf.transpose(self.alphas_L2[:,:,:,None,None]) * 
			tf.transpose(self.basis_L2[None,None,:,:,:,:])
			,axis = 2)
		l2b = self._conv_layer_3d(
				input=l1,
				kernel=w_L2,
				stride=[1,1,1,1,1],
				padding='SAME',
				name='ConvLayer2')
		print(l2b.get_shape())
		
		l2 = tf.nn.max_pool3d(l2b, ksize=[1,3,3,2,1], strides=[1,2,2,2,1], padding="VALID")
		print(l2.get_shape())
					
#		keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
#		l2 = tf.nn.dropout(l2, 1-self.dropout_rate_conv)

		# ==== Layer 3 ====		
		w_L3 = tf.reduce_sum(
			tf.transpose(self.alphas_L3[:,:,:,None,None]) * 
			tf.transpose(self.basis_L3[None,None,:,:,:,:])
			,axis = 2)
		l3b = self._conv_layer_3d(
				input=l2,
				kernel=w_L3,
				stride=[1,1,1,1,1],
				padding='SAME',
				name='ConvLayer3')
		print(l3b.get_shape())
		
		l3 = tf.nn.max_pool3d(l3b, ksize=[1,3,3,2,1], strides=[1,2,2,2,1], padding="VALID")
		print(l3.get_shape())
		
		# ==== Layer 4 ====		
		with tf.variable_scope('Flatten'):
			fshape = l3.get_shape()
			dim = fshape[1].value*fshape[2].value*fshape[3].value*fshape[4].value
			l4 = tf.reshape(l3, [-1, dim])
		print(l4.get_shape())		

#		keep_prob = tf.select(self.is_training, 1-self.dropout_rate_conv, 1)
		#pyx = tf.nn.dropout(l4, 1-self.dropout_rate_hidden)
		
		with tf.variable_scope('FullLayer'):		
			pyx = tf.matmul(l4, self.w_L4)			
			pyx = batch_norm_wrapper(pyx, self.is_training, False)
		
		return pyx	

	
		
	def _activation_summary(self, x):
		tensor_name = x.op.name
		tf.summary.histogram(tensor_name, x)
		
	def _variable_summaries(self, var, name):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + var.op.name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.scalar_summary('stddev/' + var.op.name, stddev)
			tf.scalar_summary('max/' + var.op.name, tf.reduce_max(var))
			tf.scalar_summary('min/' + var.op.name, tf.reduce_min(var))
			tf.histogram_summary(var.op.name, var)
