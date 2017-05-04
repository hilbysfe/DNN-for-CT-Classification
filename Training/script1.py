import numpy as np
import tensorflow as tf

def keep_top_k(x, k):	
	flush = x.get_shape()[3].value - k
	values, indices = tf.nn.top_k(-x, flush)
	
	a = np.zeros((x.get_shape()[0].value, x.get_shape()[1].value, x.get_shape()[2].value, len(x.get_shape().as_list())-1))
	for i in range(x.get_shape()[0].value):
		for j in range(x.get_shape()[1].value):
			for h in range(x.get_shape()[2].value):
				a[i,j,h,:] = [i,j,h]
	a = tf.convert_to_tensor(a, dtype=tf.int32)
	
	full_indices = tf.concat(3, [a, tf.expand_dims(indices[:,:,:,0], 3)])
	for i in range(1, flush):
		full_indices = tf.concat(3, [full_indices, 
									 tf.concat(3, [a, tf.expand_dims(indices[:,:,:,i], 3)])]
								)
	full_indices2 = tf.reshape(full_indices, [-1, len(x.get_shape().as_list())])
	
	to_substract = tf.sparse_to_dense(full_indices2, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
	
	return x + to_substract
	
def spatial_sparsity(x):
	return keep_top_k(x, 1)
	
def lifetime_sparsity(x, lifetime):
	
	net = tf.reshape(x, [x.get_shape()[0].value, -1])
	net = tf.transpose(net, [1, 0])
	net, indices = tf.nn.top_k(net, lifetime)
	net = tf.reshape(net, [-1, x.get_shape()[1].value, x.get_shape()[2].value, x.get_shape()[3].value])
	
	return net
	
def _WTA_layer(input, lifetime='None'):
	net = spatial_sparsity(input)
	
	if lifetime != 'None':
		net = lifetime_sparsity(net, lifetime)
	
	return net


x = tf.constant([
				[[[11., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[10., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[12., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]]],
				
				[[[7., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[8., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[9., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]]],
				
				[[[5., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[6., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]],
				[[4., 2., 1., 0.], [6., 2., 1., 0.], [6., 2., 1., 0.]]]
				
				])


sess = tf.InteractiveSession()

print(sess.run(x))
print(sess.run(_WTA_layer(x)))
# print(sess.run(tf.transpose(_WTA_layer(x), [3, 1, 2, 0])))
print(sess.run(_WTA_layer(x, 2)))