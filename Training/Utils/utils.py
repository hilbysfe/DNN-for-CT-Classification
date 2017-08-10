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


def read_dataset(datapath, labelpath, test_ratio=0.15, dense_labels=False):
	"""
	Function to read up images and labels.
	Store only paths as images wouldn't fit to memory.

	MRS@90 hardcoded as label attribute in label_filename -> column 'DF'
	"""
		
	followid_attribute = 'A2:A1489'
	label_attribute = 'DF2:DF1489'

	# --- Retrieve all patients we have images from ---
	patients = os.listdir(datapath)

	# --- Load labels from file ---
	labels_wb = ox.load_workbook(labelpath)
	labels_ws = labels_wb['Registrydatabase']

	label_dict = {key[0].value: value[0].value 
		for (key, value) in zip(labels_ws[followid_attribute], labels_ws[label_attribute]) }

	# --- Store images + labels ---
	all_images = [os.path.join(root, name)
							  for root, dirs, files in os.walk(datapath)
							  for name in files if name.endswith(".mha")
									if name.split('.')[0] in label_dict.keys()]
	patients = [ name for name in patients if name in label_dict.keys() ]	
	all_labels = [ label_dict[name] for name in patients ]				

	all_labels = np.array( all_labels )
	all_images = np.array( all_images )

	uni, classes = np.unique(all_labels, return_counts = True)
	num_classes = np.shape(classes)[0]

	num_examples = np.min(classes) * num_classes
	print('Balanced set contains %i patients' % num_examples)
	smallest_class = dict(zip(classes, uni))[np.min(classes)]

	indices = []
	for i in range(num_classes):
		indices_class_i = np.where(all_labels == i)[0]
		if i != smallest_class:
			perm = np.arange(len(indices_class_i))
			np.random.shuffle(perm)
			indices_class_i = indices_class_i[perm[:np.min(classes)]]

		indices = np.append(indices, indices_class_i)

	# Shuffle
	indices = [int(x) for x in indices]
	np.random.shuffle(indices)
	all_labels = all_labels[indices]
	all_images = all_images[indices]
	all_patients = np.array(patients)[indices]

	if not dense_labels:
		all_labels = dense_to_one_hot(all_labels, 2)
	
	# Calculate and cut the subsets
	test_size = int(num_examples*test_ratio)
	
	training_patIDs = all_patients[test_size:]
	test_patIDs = all_patients[:test_size]
	
	training_points = dict()
	test_points = dict()		
	for image in all_images:
		pat = image.split("/")[-1].split('.')[0]
		if pat in training_patIDs:
			training_points[image] = label_dict[pat]
		else:
			test_points[image] = label_dict[pat]
	
	return training_points, test_points


def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

	
	
def split_dataset(datapath, labelpath, output_path):

	training_points, test_points = read_dataset(datapath, labelpath)
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	np.save(os.path.join(output_path, 'training_points.npy'), training_points)
	np.save(os.path.join(output_path, 'test_points.npy'), test_points)
	
	return training_points, test_points	
	
class DataSet(object):

	def __init__(self, training_images, training_labels, test_images, test_labels, cross_validation_folds=0):
		
		self._Test = SubSet(test_images, test_labels)
		if cross_validation_folds == 0:
			self._Training = SubSet(training_images, training_labels)
			
			images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(file)) for file in self._Training.images])
			mean = np.mean(images, axis=0)
			std = np.std(images, axis=0)
			
			self._Training.setNormalizationParameters(mean, std)
			self._Test.setNormalizationParameters(mean, std)
			
		else:
			self._current_fold = 0
			self._fold_size = np.floor(len(training_images)/cross_validation_folds)
			
			self._image_folds = []
			self._label_folds = []
			offset = 0
			for i in range(cross_validation_folds-1):
				self._image_folds.append(training_images[offset:self._fold_size])
				self._label_folds.append(training_labels[offset:self._fold_size])
			
			self._image_folds.append(training_images[offset:])
			self._label_folds.append(training_labels[offset:])

			imageset = []
			labelset = []
			for i, fold in enumerate(self._image_folds):
				if i != self._current_fold:
					imageset += fold
			for i, fold in enumerate(self._label_folds):
				if i != self._current_fold:
					labelset += fold
			self._Training = Subset(np.array(imageset), np.array(labelset))				
			self._Validation = Subset(np.array(self._image_folds[self._current_fold]),
									np.array(self._label_folds[self._current_fold]))
	
			images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(file)) for file in self._Training.images])
			mean = np.mean(images, axis=0)
			std = np.std(images, axis=0)
			
			self._Training.setNormalizationParameters(mean, std)
			self._Validation.setNormalizationParameters(mean, std)
			
	
	def next_fold():
		self._current_fold += 1
		
		imageset = []
		labelset = []
		for i, fold in enumerate(self._image_folds):
			if i != self._current_fold:
				imageset += fold
		for i, fold in enumerate(self._label_folds):
			if i != self._current_fold:
				labelset += fold
		self._Training = Subset(np.array(imageset), np.array(labelset))	
		self._Validation = Subset(np.array(self._image_folds[self._current_fold]),
								np.array(self._label_folds[self._current_fold]))
	
		images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(file)) for file in self._Training.images])
		mean = np.mean(images, axis=0)
		std = np.std(images, axis=0)
		
		self._Training.setNormalizationParameters(mean, std)
		self._Validation.setNormalizationParameters(mean, std)
	
		return self._Training, self._Validation
	
	
	@property
	def Training(self):
		return self._Training

	@property
	def Validation(self):
		return self._Validation
	
	@property
	def Test(self):
		return self._Test

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

	@property
	def epochs_completed(self):
		return self._epochs_completed
		
	def setNormalizationParameters(mean, std):
		self._mean = mean
		self._std = std

	def next_batch(self, batch_size):
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

		image_batch = np.array([ self.getImageArray(image_path) for image_path in self._images[start:end]])
		label_batch = self._labels[start:end]

		image_batch = np.swapaxes(image_batch, 1, 2)
		image_batch = np.swapaxes(image_batch, 2, 3)

		# --- Only in case of 3D model ---
		image_batch = np.expand_dims(image_batch, axis=4)			

		return image_batch, label_batch


	def getImageArray(self, image_path):
		"""
		Returns:
			Numpy array of the loaded image
		Args:
			image_path: Path of image to read from.
		"""
		return div0(sitk.GetArrayFromImage(sitk.ReadImage(image_path)) - self._mean, self._std)[9:25,:,:]
		
		
