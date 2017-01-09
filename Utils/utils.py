from __future__ import division

import os
import SimpleITK as sitk
import numpy as np

def read_datasets(images_root_path, subfolder_name, label_path, validation_ratio):
	"""
	Function to read up images and labels.
	Store only paths as images would fit to memory.
	"""
	patients = [name for name in os.listdir(images_root_path)
            if os.path.isdir(os.path.join(images_root_path, name))]
	# subfolder_name = "\\CT24h\\thick\\"
	
	all_samples = len(patients)
	
	all_data = []
	labels = np.zeros(all_samples)
	
	for name in patients:
		sample_filepath = images_root_path + subfolder_name + "pat" + name.zfill(4) + ".mhd"
		all_data.append(filepath)
		
	with open(label_path, 'rb') as f:
		label_list = f.readlines()
	
	for i in range(all_samples):
		labels[i] = int(label_list[i])

	perm = np.arange(all_samples)
	np.random.shuffle(perm)
	
	validation_size = all_samples*validation_ratio
	
	training_images = np.empty(shape=[all_samples-validation_size], dtype=string)
	training_labels = np.zeros(all_samples-validation_size)
	validation_images = np.empty(shape=[validation_size], dtype=string)
	validation_labels = np.zeros(validation_size)
	
	for i in range(all_samples):
		if i < validation_size:
			validation_images[i] = all_data[perm[i]]
			validation_labels[i] = labels[perm[i]]
		else:
			training_images[i] = all_data[perm[i]]
			training_labels[i] = labels[perm[i]]
	
	return DataSet(training_images, training_labels), DataSet(validation_images, validation_labels)
		
		

class DataSet(object):
	"""
	Utility class to handle dataset structure.
	"""

	def __init__(self, images, labels):
		"""
		Builds dataset with images and labels.
		Args:
			images: Images data.
			labels: Labels data
		"""
		assert images.shape[0] == labels.shape[0], (
					"images.shape: {0}, labels.shape: {1}".format(str(images.shape), str(labels.shape)))

		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size):
		"""
		Return the next `batch_size` examples from this data set.
		Args:
			batch_size: Batch size.
		"""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1

			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch
		
		image_batch = np.array([ \
			sitk.GetArrayFromImage(sitk.ReadImage(image_path)) \
				for image_path in self._images[start:end]])
		label_batch = self._labels[start:end]
		
		return image_batch, label_batch
