from __future__ import division

import os
import SimpleITK as sitk
import numpy as np
import openpyxl as ox
import pickle
import tensorflow as tf

# CTA / MIP
#MIN_BOUND = 50.0
#MAX_BOUND = 300.0
# NCCT
MIN_BOUND = 0.0
MAX_BOUND = 100.0


def online_flattened_mean(files):
	# Calculates mean of all pixels in the dataset, considering the intensities normalized
	mean = 0
	for file in files:
		data = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(file)))
		mean += (np.sum(data) / (len(files) * np.shape(data)[0] * np.shape(data)[1]))

	return mean

def online_flattened_std(files, mean):
	# Calculates mean of all pixels in the dataset, considering the intensities normalized
	std = 0
	for file in files:
		data = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(file)))
		std += (np.sum(np.square(data)) / (len(files) * np.shape(data)[0] * np.shape(data)[1]))

	return np.sqrt(std-np.square(mean))

def online_flattened_mean_3d(files):
	# Calculates mean of all pixels in the dataset, considering the intensities normalized
	mean = 0
	for file in files:
		data = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(file)))
		mean += (np.sum(data) / (len(files) * np.shape(data)[0] * np.shape(data)[1] * np.shape(data)[2]))

	return mean

def online_flattened_std_3d(files, mean):
	# Calculates mean of all pixels in the dataset, considering the intensities normalized
	std = 0
	for file in files:
		data = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(file)))
		std += (np.sum(np.square(data)) / (len(files) * np.shape(data)[0] * np.shape(data)[1] * np.shape(data)[2]))

	return np.sqrt(std-np.square(mean))

def normalize_image(data):
	data = (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	data[data > 1] = 1
	data[data < 0] = 0

	return data


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

def split_datasets(datapath, labelpath, output, label_name='', val_folds=4, val_ratio=0.1):
	"""
	Function to read up images and labels.
	Store only paths as images wouldn't fit to memory.

	"""
	attribute_dict = {
		'collaterals': 'C2:C1489',
		'collaterals_imp': 'E2:E1489',
		'tici': 'G2:G1489',
		'tici_imp': 'I2:I1489',
		'nihss': 'K2:K1489',
		'nihss_imp': 'M2:M1489',
		'mrs': 'N2:N1489',
		'aspects': 'R2:R1489',
		'aspects_imp': 'S2:S1489'
	}

	if 'affected' in labelpath:
		followid_attribute = 'A1:A1527'
		label_attribute = 'B1:B1527'
		label_name = "affected_side"
	else:
		followid_attribute = 'A2:A1489'
		label_attribute = attribute_dict[label_name]

	# --- Retrieve all patients we have images from ---
	patients = os.listdir(datapath)

	# --- Load labels from file ---
	labels_wb = ox.load_workbook(labelpath)
	labels_ws = labels_wb['Registrydatabase']

	# USE THIS FOR ALL IAT DEPENDENT LABELS
	label_dict = {key[0].value: value[0].value
				  for i, (key, value) in enumerate(zip(labels_ws[followid_attribute], labels_ws[label_attribute]))
				  if value[0].value is not None and value[0].value is not '' and labels_ws['O' + str(i + 2)].value == 3}

	# USE THIS FOR COLLATERALS AND AFFECTED SIDE
#	label_dict = {key[0].value: value[0].value
#				  for i, (key, value) in enumerate(zip(labels_ws[followid_attribute], labels_ws[label_attribute]))
#				  if value[0].value is not None and value[0].value is not ''}



	# --- Split regarding classes ---
	class0_images = [os.path.join(root, name)
					 for root, dirs, files in os.walk(datapath)
					 for name in files if name.endswith(".mha")
					 if name.split('.')[0] in label_dict.keys() and
					 label_dict[name.split('.')[0]] == 0]
	class1_images = [os.path.join(root, name)
					 for root, dirs, files in os.walk(datapath)
					 for name in files if name.endswith(".mha")
					 if name.split('.')[0] in label_dict.keys() and
					 label_dict[name.split('.')[0]] == 1]

	num_examples = 2 * min(len(class0_images), len(class1_images))

	# --- Schuffle both classes ---
	perm = np.arange(len(class0_images))
	np.random.shuffle(perm)
	class0_images = np.array(class0_images)[perm]

	perm = np.arange(len(class1_images))
	np.random.shuffle(perm)
	class1_images = np.array(class1_images)[perm]

	# --- Balance out ---
	balanced_size = min(len(class0_images), len(class1_images))

	class0_images = class0_images[:balanced_size]
	class1_images = class1_images[:balanced_size]

	# --- Split into balanced test-set, training-set ---
	# --- Prepare folds ---
	image0_folds = []
	image1_folds = []

	offset = 0
	point_per_fold = int(balanced_size / val_folds)
	val_size = int((balanced_size - point_per_fold)*val_ratio)
	for i in range(val_folds):
		image0_folds.append(
			class0_images[offset:offset + point_per_fold]
		)
		image1_folds.append(
			class1_images[offset:offset + point_per_fold]
		)

		offset += point_per_fold

	offset = 0
	for current_fold in range(val_folds):
		# --- Init sets with 1st fold ---
		training_folds_imgs0 = np.concatenate(
			[fold for i, fold in enumerate(image0_folds) if i != current_fold])
		test_imgset0 = image0_folds[current_fold]

		training_folds_imgs1 = np.concatenate(
			[fold for i, fold in enumerate(image1_folds) if i != current_fold])
		test_imgset1 = image1_folds[current_fold]

		# ---- Shuffle training folds + split into train-val -----
		perm = np.arange(len(training_folds_imgs0))
		np.random.shuffle(perm)
		training_folds_imgs0 = np.array(training_folds_imgs0)[perm]

		perm = np.arange(len(training_folds_imgs1))
		np.random.shuffle(perm)
		training_folds_imgs1 = np.array(training_folds_imgs1)[perm]

		validation_imgset0 = training_folds_imgs0[:val_size]
		validation_imgset1 = training_folds_imgs1[:val_size]

		training_imgset0 = training_folds_imgs0[val_size:]
		training_imgset1 = training_folds_imgs1[val_size:]

		training_points = dict(
			zip(np.concatenate((training_imgset0, training_imgset1)),
				np.concatenate((np.zeros((len(training_imgset0),), dtype=np.int),
								np.ones((len(training_imgset1),), dtype=np.int)))))
		validation_points = dict(
			zip(np.concatenate((validation_imgset0, validation_imgset1)),
				np.concatenate((np.zeros((len(validation_imgset0),), dtype=np.int),
								np.ones((len(validation_imgset1),), dtype=np.int)))))
		test_points = dict(
			zip(np.concatenate((test_imgset0, test_imgset1)),
				np.concatenate((np.zeros((len(test_imgset0),), dtype=np.int),
								np.ones((len(test_imgset1),), dtype=np.int)))))

		# ------- Save -------
		tf.gfile.MakeDirs(os.path.join(output, label_name, str(current_fold)))

		with open(os.path.join(output, label_name, str(current_fold), 'training_points.npy'), 'wb') as handle:
			pickle.dump(training_points, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(output, label_name, str(current_fold), 'validation_points.npy'), 'wb') as handle:
			pickle.dump(validation_points, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(output, label_name, str(current_fold), 'test_points.npy'), 'wb') as handle:
			pickle.dump(test_points, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_dataset(datapath, val_folds=4):
	training_points_list = []
	test_points_list = []
	validation_points_list = []
	for i in range(val_folds):
		with open(os.path.join(datapath, str(i), 'training_points.npy'), 'rb') as handle:
			training_points = pickle.load(handle)
		training_points_list.append(training_points)

		with open(os.path.join(datapath, str(i), 'test_points.npy'), 'rb') as handle:
			test_points = pickle.load(handle)
		test_points_list.append(test_points)

		with open(os.path.join(datapath, str(i), 'validation_points.npy'), 'rb') as handle:
			validation_points = pickle.load(handle)
		validation_points_list.append(validation_points)

	return training_points_list, test_points_list, validation_points_list


class DataSet(object):

	def __init__(self, training_points_list, test_points_list, validation_points_list, normalize=False, img3d=False):
		print('Init Dataset...')
		self.img3d = img3d
		self._current_fold = 0
		self._Test = []
		self._Training = []
		self._Validation = []
		self._folds = len(training_points_list)
		for i in range(len(training_points_list)):
			training_points = training_points_list[i]
			test_points = test_points_list[i]
			validation_points = validation_points_list[i]

			# === TEST-SET ===
			# --- Split and shuffle ---
			perm = np.arange(int(len(test_points) / 2))
			np.random.shuffle(perm)
			test_images0 = np.array([image for image in test_points if test_points[image] == 0])[perm]
			np.random.shuffle(perm)
			test_images1 = np.array([image for image in test_points if test_points[image] == 1])[perm]

			test_labels0 = np.array(
				[np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(test_images0))])
			test_labels1 = np.array(
				[np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(test_images1))])

			self._Test.append(SubSet(test_images0, test_images1, test_labels0, test_labels1))

			# === TRAINING-SET ===
			# --- Split and shuffle ---
			perm = np.arange(int(len(training_points)/2))
			np.random.shuffle(perm)
			training_images0 = np.array([image for image in training_points if training_points[image] == 0])[perm]
			np.random.shuffle(perm)
			training_images1 = np.array([image for image in training_points if training_points[image] == 1])[perm]

			training_labels0 = np.array(
				[np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(training_images0))])
			training_labels1 = np.array(
				[np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(training_images1))])

			self._Training.append(SubSet(training_images0, training_images1, training_labels0, training_labels1))

			# === VALIDATION-SET ===
			perm = np.arange(int(len(validation_points)/2))
			np.random.shuffle(perm)
			validation_images0 = np.array([image for image in validation_points if validation_points[image] == 0])[perm]
			np.random.shuffle(perm)
			validation_images1 = np.array([image for image in validation_points if validation_points[image] == 1])[perm]

			validation_labels0 = np.array(
				[np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(validation_images0))])
			validation_labels1 = np.array(
				[np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(validation_images1))])

			self._Validation.append(SubSet(validation_images0, validation_images1, validation_labels0, validation_labels1))

			print('Creating folds...done.')

			# --- prepare normalization ---
			if normalize:
				print('Computing mean...')
				if not self.img3d:
					mean = online_flattened_mean(self._Training[i].images)
					std = online_flattened_std(self._Training[i].images, mean)
				else:
					mean = online_flattened_mean_3d(self._Training[i].images)
					std = online_flattened_std_3d(self._Training[i].images, mean)
				print('Computing mean...done.')

				self.Normalization = True
				self._Training[i].Normalization = True
				self._Test[i].Normalization = True
				self._Validation[i].Normalization = True
				self._Training[i].setNormalizationParameters(mean, std)
				self._Test[i].setNormalizationParameters(mean, std)
				self._Validation[i].setNormalizationParameters(mean, std)
			else:
				self.Normalization = False
				self._Training[i].Normalization = False
				self._Validation[i].Normalization = False
				self._Test[i].Normalization = False

		print('Init Dataset...done.')

	def next_fold(self):
		self._current_fold += 1

	def reset(self):
		self._current_fold = 0

	@property
	def Training(self):
		return self._Training[self._current_fold]

	@property
	def Validation(self):
		return self._Validation[self._current_fold]

	@property
	def Test(self):
		return self._Test[self._current_fold]

class SubSet(object):
	"""
	Utility class to handle training and validation set structure.
	"""

	def __init__(self, images0, images1, labels0, labels1):
		"""
		Builds dataset with images and labels.
		Args:
			images0: Images data of class0.
			labels0: Labels data of class0.
		"""
		assert images0.shape[0] == labels0.shape[0], (
			"images.shape: {0}, labels.shape: {1}".format(str(images0.shape), str(labels0.shape)))
		assert images1.shape[0] == labels1.shape[0], (
			"images.shape: {0}, labels.shape: {1}".format(str(images1.shape), str(labels1.shape)))

		self._num_examples = images0.shape[0]
		self._images = np.concatenate((images0, images1))
		self._labels = np.concatenate((labels0, labels1))
		self._epochs_completed = 0
		self._index_in_epoch = 0

		perm = np.arange(self._num_examples)
		np.random.shuffle(perm)

		self._images0 = images0[perm]
		self._labels0 = labels0[perm]

		self._images1 = images1[perm]
		self._labels1 = labels1[perm]

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

	def reset(self):
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def setNormalizationParameters(self, mean, std):
		self._mean = mean
		self._std = std

	def next_batch(self, batch_size, bases3d=False):
		"""
		Return the next `batch_size` examples from this data set.
		Args:
			batch_size: Batch size.
		"""
		assert batch_size <= 2 * self._num_examples

		start = self._index_in_epoch
		self._index_in_epoch += int(batch_size / 2)
		if self._index_in_epoch >= self._num_examples:
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images0 = self._images0[perm]
			self._images1 = self._images1[perm]

			start = 0
			self._index_in_epoch = int(batch_size / 2)

		end = self._index_in_epoch

		perm = np.arange(batch_size)
		np.random.shuffle(perm)

		label_batch = np.concatenate((self._labels0[start:end], self._labels1[start:end]))[perm]
		image_batch = np.array(
			[self.getImageArray(image0) for image0 in self._images0[start:end]] +
			[self.getImageArray(image1) for image1 in self._images1[start:end]]
		)[perm]

		# --- Only in case of 3D model ---
		if bases3d:
			image_batch = np.expand_dims(image_batch, axis=4)
			image_batch = np.swapaxes(image_batch, 1, 2)
			image_batch = np.swapaxes(image_batch, 2, 3)
		else:
			image_batch = np.expand_dims(image_batch, axis=3)

		return image_batch, label_batch

	def getImageArray(self, image_path):
		"""
		Returns:
			Numpy array of the loaded image
		Args:
			image_path: Path of image to read from.
		"""
		if self.Normalization:
			return np.divide(normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(image_path))) - self._mean, self._std)
		else:
			sl = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
			return sl
