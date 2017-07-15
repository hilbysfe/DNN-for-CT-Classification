import tensorflow as tf

x = tf.constant([[6., 2., 1.], [1., 4., 5.]])
print(x.get_shape())
k = 2
values, indices = tf.nn.top_k(-x, k)
print(indices.get_shape())

my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
my_range_repeated = tf.tile(my_range, [1, k])
print(my_range_repeated.get_shape())

a = tf.expand_dims(my_range_repeated, 2)
b = tf.expand_dims(indices, 2)

full_indices = tf.concat(2, [a, b])
print(full_indices.get_shape())
full_indices2 = tf.reshape(full_indices, [-1, 2])
print(full_indices2.get_shape())


to_substract = tf.sparse_to_dense(full_indices2, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
print(to_substract.get_shape())

res = x + to_substract

sess = tf.InteractiveSession()

print(sess.run(x))
print(sess.run(indices))
print(sess.run(my_range_repeated))
print(sess.run(a))
print(sess.run(b))
print(sess.run(full_indices))
print(sess.run(full_indices2))
print(sess.run(to_substract))
print(sess.run(res))