import tensorflow as tf
import numpy as np

N_SAMPLES = 100
NUM_THREADS = 4
NUM_GPUS=4
BATCH_SIZE=2

def next_batch(data, labels):
	return [ data[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in np.arange(NUM_GPUS)], \
		[ labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in np.arange(NUM_GPUS)]

def train():
	with tf.device('/cpu:0'):
		# === LOAD DATA ===
		# Generating some simple data
		# create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
		data = 10 * np.random.randn(N_SAMPLES, 4) + 1
		targets = np.random.randint(0, 2, size=N_SAMPLES)

		# === DEFINE FEED_DICT ===
		def feed_dict():
			xs, ys = next_batch(data, targets)
			return {x: xs, y: ys}

		# === DEFINE PLACHOLDERS ===
		x = tf.placeholder(tf.float32, [NUM_GPUS, BATCH_SIZE, 4], name='x-input')
		y = tf.placeholder(tf.float32, [NUM_GPUS, BATCH_SIZE], name='y-input')

		# === DEFINE QUEUE OPS ===
		queue = tf.FIFOQueue(capacity=NUM_GPUS, dtypes=[tf.float32, tf.float32], shapes=[(BATCH_SIZE, 4), (BATCH_SIZE)])
		enqueue_op = queue.enqueue_many([x, y])
		out = []
		for i in np.arange(NUM_GPUS):
			data_batch, label_batch = queue.dequeue()	
			with tf.device('/gpu:%d' % i):	
				out.append(tf.multiply(tf.transpose(data_batch), tf.cast(label_batch, tf.float32)))

		avg_out = tf.reduce_mean(out)

		close_op = queue.close()

		with tf.Session() as sess:
			# Create a coordinator, launch the queue runner threads.
			coord = tf.train.Coordinator()
			tf.train.start_queue_runners(sess, coord=coord)
			for i in np.arange(2):
				sess.run(enqueue_op, feed_dict())

				if coord.should_stop():
					break

				o = sess.run(avg_out, feed_dict())
				print(str(o))
		
			coord.request_stop()
			coord.join()
			sess.run(close_op)


if __name__ == '__main__':
	train()
