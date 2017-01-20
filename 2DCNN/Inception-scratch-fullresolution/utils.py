from __future__ import division

import os
import SimpleITK as sitk
import numpy as np
import openpyxl as ox

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

def preprocess_data(X_raw):

	X = X_raw.copy()

	# Substract the mean
	mean_image = np.mean(X)
	X -= mean_image

	return X

def read_dataset(image_dir, label_filename, validation_ratio=0.3, preprocess=False):
	"""
	Function to read up images and labels.
	Store only paths as images would fit to memory.

	MRS@90 hardcoded as label attribute in label_filename -> column 'AE'
	"""

	followid_attribute = 'AB2:AB501'
	label_attribute = 'AE2:AE501'

	# --- Retrieve all patients we have images from ---
	patients = [name for name in os.listdir(image_dir) if name.split('.', 1)[1]=="mhd"]

	all_samples = len(patients)

	# --- Load labels from file ---
	labels_wb = ox.load_workbook(label_filename)
	labels_ws = labels_wb.active

	label_dict = {key[0].value.zfill(4): value[0].value 
		for (key, value) in zip(labels_ws[followid_attribute], labels_ws[label_attribute]) }

	# --- Store images(optionally more per patient) + labels ---
	all_images = [ image_dir + '\\' + name for name in patients ]
	all_labels = [ label_dict[name.split('.',1)[0].replace('_2', '')] for name in patients ]				

	all_labels = np.array( all_labels )
	all_images = np.array( all_images )

	# --- Preprocess if desired ---
	if preprocess:
		all_images = preprocess_data(all_images)

	num_classes = len(np.unique(all_labels))
	
	all_labels = dense_to_one_hot(all_labels, num_classes)

	# --- Return DataSet objects ---	
	return DataSet(all_images, all_labels, validation_ratio)

class DataSet(object):
	"""
	Utility class to handle complete dataset structure.
	"""
	def initialize_subsets(self):
		
		self._epochs_completed += 1
		
		# --- Shuffle examples to split fairly ---	
		perm = np.arange(np.shape(self._images)[0])
		np.random.shuffle(perm)

		self._images = self._images[perm]
		self._labels = self._labels[perm]

		# --- Split data to training and validation set ---
		validation_images = self._images[:self._validation_size]
		validation_labels = self._labels[:self._validation_size]
		training_images = self._images[self._validation_size:]
		training_labels = self._labels[self._validation_size:]
		
		self._Training = SubSet(training_images, training_labels)
		self._Validation = SubSet(validation_images, validation_labels)
		
	def __init__(self, images, labels, validation_ratio):
		"""
		Builds dataset with images and labels.
		Args:
			images: Images data.
			labels: Labels data
		"""
		assert images.shape[0] == labels.shape[0], (
			"images.shape: {0}, labels.shape: {1}".format(str(images.shape), str(labels.shape)))

		self._images = images
		self._labels = labels
				
		self._epochs_completed = 0
		self._validation_size = int(np.shape(self._images)[0]*validation_ratio)

		self.initialize_subsets()		
		self._epochs_completed = 0
		
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
		if self._index_in_epoch > self._num_examples:
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

			image_batch = np.expand_dims(image_batch, axis=4)
		else:
			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)
			image_batch = np.swapaxes(image_batch, 3, 4)
			
			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)

		return image_batch, label_batch


	def getImageArray(self, image_path, multiAttenuation):
		"""
		Returns 
		Args:
			image_path: Path of image to read from.
			multiAttenuation: Boolean whether ranges of attenuation is required
		"""
		
		if not multiAttenuation:
			return sitk.GetArrayFromImage(sitk.ReadImage(image_path))[9:25,16:240,16:240]
		else:
			arr = sitk.GetArrayFromImage(sitk.ReadImage(image_path))[9:25,16:240,16:240]
			return np.array([
				np.multiply(np.divide(np.add(arr, 50),130),255),
				np.multiply(np.divide(arr, 80),255),
				np.multiply(np.divide(arr, 130),255)
			])