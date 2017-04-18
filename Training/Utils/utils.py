from __future__ import division

import os
import SimpleITK as sitk
import numpy as np
import openpyxl as ox
import operator

def dense_to_one_hot(labels_dense, num_classes):
	"""
	Convert class labels from scalars to one-hot vectors.
	Args:
	labels_dense: Dense labels.
	num_classes: Number of classes.

	Outputs:
	labels_one_hot: One-hot encoding for labels.
	"""

	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

	return labels_one_hot

def read_dataset(image_dir, label_filename, dense_labels=False):
	"""
	Function to read up images and labels.
	Store only paths as images would fit to memory.

	MRS@90 hardcoded as label attribute in label_filename -> column 'AE'
	"""

	followid_attribute = 'AB2:AB501'
	label_attribute = 'AE2:AE501'

	# --- Retrieve all images ---
	training_images = [image_dir + '\\Training\\' + name \
				for name in os.listdir(image_dir + '\\Training\\') if name.split('.', 1)[1]=="mhd"]
	validation_images = [image_dir + '\\Validation\\' + name \
				for name in os.listdir(image_dir + '\\Validation\\') if name.split('.', 1)[1]=="mhd"]
	test_images = [image_dir + '\\Test\\' + name \
				for name in os.listdir(image_dir + '\\Test\\') if name.split('.', 1)[1]=="mhd"]

	# --- Load labels from file ---
	labels_wb = ox.load_workbook(label_filename)
	labels_ws = labels_wb.active

	label_dict = {key[0].value.zfill(4): value[0].value 
		for (key, value) in zip(labels_ws[followid_attribute], labels_ws[label_attribute]) }

	# --- Retrieve labels ---
	training_labels = [ label_dict[name.split("\\")[-1].split('.')[0].split('_')[0]] for name in training_images ]
	validation_labels = [ label_dict[name.split("\\")[-1].split('.')[0].split('_')[0]] for name in validation_images ]
	test_labels = [ label_dict[name.split("\\")[-1].split('.')[0].split('_')[0]] for name in test_images ]
		
	# Shuffling
	perm = np.arange(len(training_images))
	np.random.shuffle(perm)
	training_images = np.array(training_images)[perm]
	training_labels = np.array(training_labels)[perm]
		
	perm = np.arange(len(validation_images))
	np.random.shuffle(perm)
	validation_images = np.array(validation_images)[perm]
	validation_labels = np.array(validation_labels)[perm]
	
	perm = np.arange(len(test_images))
	np.random.shuffle(perm)
	test_images = np.array(test_images)[perm]
	test_labels = np.array(test_labels)[perm]

	if not dense_labels:
		training_labels = dense_to_one_hot(training_labels, 2)
		validation_labels = dense_to_one_hot(validation_labels, 2)
		test_labels = dense_to_one_hot(test_labels, 2)

	# --- Return DataSet objects ---	
	return DataSet(training_images, training_labels, validation_images, validation_labels, test_images, test_labels)

class DataSet(object):

	def __init__(self, training_images, training_labels, validation_images, validation_labels, test_images, test_labels):
		
		self._Training = SubSet(training_images, training_labels)
		self._Validation = SubSet(validation_images, validation_labels)
		self.Test = SubSet(test_images, test_labels)

	@property
	def epochs_completed(self):
		return self._epochs_completed

	@property
	def Training(self):
		return self._Training

	@property
	def Validation(self):
		return self._Validation

class SubSet(object):
	"""
	Utility class to handle training and validation set structure.
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
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def index_in_epoch(self):
		return self._index_in_epoch


	def next_batch(self, batch_size, multiAttenuation = False):
		"""
		Return the next `batch_size` examples from this data set.
		Args:
						batch_size: Batch size.
		"""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch >= self._num_examples:
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]

			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch

		image_batch = np.array([ self.getImageArray(image_path, multiAttenuation) for image_path in self._images[start:end]])
		label_batch = self._labels[start:end]

		if not multiAttenuation:
			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)

			# --- Only in case of 3D model ---
			# image_batch = np.expand_dims(image_batch, axis=4)
			
		else:
			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)
			image_batch = np.swapaxes(image_batch, 3, 4)

			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)

		return image_batch, label_batch


	def getImageArray(self, image_path, multiAttenuation=False):
		"""
		Returns 
		Args:
			image_path: Path of image to read from.
			multiAttenuation: Boolean whether ranges of attenuation is required
		"""

		if not multiAttenuation:
			return sitk.GetArrayFromImage(sitk.ReadImage(image_path))[9:25,:,:]
		else:
			arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))[4:20,:,:]
			return np.array([
				np.multiply(np.divide(np.add(arr, 50),130),255),
				np.multiply(np.divide(arr, 80),255),
				np.multiply(np.divide(arr, 130),255)
			])