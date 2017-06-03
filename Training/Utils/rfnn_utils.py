import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import convolve1d as conv
from Utils.cnn_utils import batch_norm_wrapper 

def floatX(X):
	return np.asarray(X, dtype=np.float32)	

def _rfnn_conv_layer_2d_with_alphas(input, basis, alphas, biases, strides, padding, is_training, bnorm=False):
	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
				tf.transpose(alphas[:,:,:,None,None]) *
				tf.transpose(basis[None,None,i,:,:,:])
					,axis=2, name='weights_' + str(i) )
		
		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
		if bnorm:
			conv = batch_norm_wrapper(conv, is_training, True)
		else:
			conv = tf.nn.bias_add(conv, biases)				
		conv_out = tf.nn.relu(conv, name='Activation')
		
		kernels.append(kernel)
		outputs.append(conv_out)
	
	return outputs, kernels
	
def _rfnn_conv_layer_2d(input, basis, omaps, strides, padding, is_training, bnorm=False):
	alphas = tf.get_variable(
		'alphas',
		shape=[omaps, input.get_shape()[-1].value, np.shape(basis)[1]],
		initializer=tf.random_uniform_initializer(-1.0, 1.0))
		
	if not bnorm:
		biases = tf.get_variable(
			'biases',
			shape=[omaps],
			initializer=tf.constant_initializer(0.0),
			dtype=tf.float32)
	
	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
				tf.transpose(alphas[:,:,:,None,None]) *
				tf.transpose(basis[None,None,i,:,:,:])
					,axis=2, name='weights_' + str(i) )
		
		conv = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
		if bnorm:
			conv = batch_norm_wrapper(conv, is_training, True)
		else:
			conv = tf.nn.bias_add(conv, biases)				
		conv_out = tf.nn.relu(conv, name='Activation')
		
		kernels.append(kernel)
		outputs.append(conv_out)
	
	return alphas, biases, outputs, kernels
	
def _rfnn_deconv_layer_2d(input, basis, omaps, oshape, strides, padding, bnorm=False):
	alphas = tf.get_variable(
		'alphas',
		shape=[input.get_shape()[-1].value, omaps, np.shape(basis)[1]],
		initializer=tf.random_uniform_initializer(-1.0, 1.0))
		
	biases = tf.get_variable(
		'biases',
		shape=[omaps],
		initializer=tf.constant_initializer(0.0),
		dtype=tf.float32)
	
	kernels = []
	outputs = []
	for i in range(np.shape(basis)[0]):
		kernel = tf.reduce_sum(
				tf.transpose(alphas[:,:,:,None,None]) *
				tf.transpose(basis[None,None,i,:,:,:])
					,axis=2, name='weights_' + str(i) )
		
		deconv = tf.nn.conv2d_transpose(input, kernel, oshape, strides, padding)
		deconv = tf.nn.bias_add(deconv, biases)	
		deconv = tf.nn.relu(deconv, name='Activation')
		
		outputs.append(deconv)
	
	return outputs
	
	
def init_basis_hermite_2D(kernel, sigmas, bases):
	nrBasis = 15
	hermiteBasis = np.empty( (np.int(np.shape(sigmas)[0]), np.int(nrBasis), np.int(kernel), np.int(kernel)) )
	
	for i, sigma in enumerate(sigmas):
		x = np.arange(-np.int((kernel-1)/2), np.int((kernel-1)/2)+1, dtype=np.float)
		impulse = np.zeros( (kernel, kernel) )
		impulse[np.int((kernel-1)/2),np.int((kernel-1)/2)] = 1.0
		
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
		
		hermiteBasis[i,0,:,:] = gauss0					# g
		hermiteBasis[i,1,:,:] = conv(gauss0y, g1, axis=1) # g_x
		hermiteBasis[i,2,:,:] = conv(gauss0x, g1, axis=0) # g_y
		hermiteBasis[i,3,:,:] = conv(gauss0y, g2, axis=1) # g_xx
		hermiteBasis[i,4,:,:] = conv(gauss0x, g2, axis=0) # g_yy
		hermiteBasis[i,5,:,:] = conv(gauss1x, g1, axis=0) # g_xy
		hermiteBasis[i,6,:,:] = conv(gauss0y, g3, axis=1) # g_xxx
		hermiteBasis[i,7,:,:] = conv(gauss0x, g3, axis=0) # g_yyy
		hermiteBasis[i,8,:,:] = conv(gauss1y, g2, axis=1) # g_xxy
		hermiteBasis[i,9,:,:] = conv(gauss1x, g2, axis=0) # g_yyx
		hermiteBasis[i,10,:,:] = conv(gauss0y, g4, axis=1) # g_xxxx
		hermiteBasis[i,11,:,:] = conv(gauss0x, g4, axis=0) # g_yyyy
		hermiteBasis[i,12,:,:] = conv(gauss1y, g3, axis=1) # g_xxxy
		hermiteBasis[i,13,:,:] = conv(gauss1x, g3, axis=0) # g_yyyx
		hermiteBasis[i,14,:,:] = conv(gauss2x, g2, axis=0) # g_yyxx

	return tf.constant(floatX(hermiteBasis[:,0:bases,:,:]))

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

def init_alphas(nrFilters,channels,nrBasis,name):
	return tf.get_variable(
		name,
		shape=[nrFilters,channels,nrBasis],
		initializer=tf.random_uniform_initializer(-1.0, 1.0))
		
def init_biases(channels,name):
	return tf.get_variable(
		name, shape=[channels],
		initializer=tf.constant_initializer(0.0),
		dtype=tf.float32)
