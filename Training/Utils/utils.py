from __future__ import division

import os
import SimpleITK as sitk
import numpy as np
import openpyxl as ox
import operator
import pickle
import tensorflow as tf
import re

MIN_BOUND = 0.0
MAX_BOUND = 200.0

def _add_loss_summaries(total_loss):

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)

def _variable_summaries(var, name):
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

def online_std(files):
    n = 0
    mean = np.zeros(np.shape(sitk.GetArrayFromImage(sitk.ReadImage(files[0]))))
    std = np.zeros(np.shape(sitk.GetArrayFromImage(sitk.ReadImage(files[0]))))

    for file in files:
        data = sitk.GetArrayFromImage(sitk.ReadImage(file))
        n += 1
        delta = np.subtract(data, mean)
        mean = np.add(mean, np.divide(delta,n))
        delta2 = np.subtract(data, mean)
        std = np.add(std, np.multiply(delta, delta2))

    if n<2:
        return float('nan')
    else:
        return mean, np.divide(std, n-1)

def online_flattened_mean(files):
    # Calculates mean of all pixels in the dataset, considering the intensities normalized
    mean = 0
    for file in files:
        data = normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(file)))
        mean += np.sum(data)

    return mean/(len(files)*np.shape(data)[0]*np.shape(data)[1]*np.shape(data)[2])

def normalize_image(data):
    data = (data-MIN_BOUND) / (MAX_BOUND-MIN_BOUND)
    data[data>1] = 1
    data[data<0] = 0

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


def read_dataset(datapath, labelpath, test_ratio=0.15):
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
    all_labels = [label_dict[name.split('.')[0]]
                              for root, dirs, files in os.walk(datapath)
                              for name in files if name.endswith(".mha")
                                    if name.split('.')[0] in label_dict.keys()]

    # --- Filter out the ones missing from Database ---
    patients = [ name for name in patients if name in label_dict.keys() ]	

    all_labels = np.array( all_labels )
    all_images = np.array( all_images )

    # --- Balance out dataset ---
    uni, classes = np.unique(all_labels, return_counts = True)
    num_classes = np.shape(classes)[0]

    num_examples = np.min(classes) * num_classes
    print('Balanced set contains %i patients' % num_examples)
    smallest_class = dict(zip(classes, uni))[np.min(classes)]

    # --- Select subsets keeping balance ---
    test_size = int(num_examples*test_ratio)
    indices = []
    test_patIDs = []
    training_patIDs = []
    for i in range(num_classes):
        indices_class_i = np.where(all_labels == i)[0]
        if i != smallest_class:
            perm = np.arange(len(indices_class_i))
            np.random.shuffle(perm)
            indices_class_i = indices_class_i[perm[:np.min(classes)]]

        indices = np.append(indices, indices_class_i)
        # --- Shuffle ---
        indices = [int(x) for x in indices]
        np.random.shuffle(indices)

        test_patIDs += list(np.array(patients)[indices[:int(test_size/num_classes)]])
        training_patIDs += list(np.array(patients)[indices[int(test_size/num_classes):]])

    training_points = dict()
    test_points = dict()	
    for image in all_images:
        pat = image.split("/")[-1].split('.')[0]
        if pat in training_patIDs:
            training_points[image] = all_labels[np.where(all_images == image)[0][0]]
        else:
            test_points[image] = all_labels[np.where(all_images == image)[0][0]]

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

    with open(os.path.join(output_path, 'training_points.npy'), 'wb') as handle:
        pickle.dump(training_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_path, 'test_points.npy'), 'wb') as handle:
        pickle.dump(test_points, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return training_points, test_points	

class DataSet(object):

    def __init__(self, training_images, training_labels, test_images, test_labels, cross_validation_folds=0, normalize=False):
        print('Init Dataset...')
        self._Test = SubSet(test_images, dense_to_one_hot(test_labels, 2))

        if cross_validation_folds == 0:
            self._Training = SubSet(training_images, dense_to_one_hot(training_labels, 2))

            if normalize:
                print('Computing mean...')
                mean, std = online_flattened_mean(self._Training.images)
                print('Computing mean...done.')

                self._Training.setNormalizationParameters(mean)
                self._Test.setNormalizationParameters(mean)
            else:
                self.Normalization = False
                self._Training.Normalization = False
                self._Validation.Normalization = False

        else:
            print('Creating folds...')

            self._current_fold = 0
            self._fold_size = len(training_labels)/cross_validation_folds
            images_0 = training_images[np.where(training_labels==0)[0]]
            images_1 = training_images[np.where(training_labels==1)[0]]
            labels_0 = np.array([ np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(images_0)) ])
            labels_1 = np.array([ np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(images_0)) ])
            self._image_folds = []
            self._label_folds = []
            offset = 0
            for i in range(cross_validation_folds-1):
                image_fold = []
                image_fold += list(images_0[offset:offset+int(self._fold_size/2)])
                image_fold += list(images_1[offset:offset+int(self._fold_size/2)])

                label_fold = np.zeros((2*int(self._fold_size/2), 2))
                label_fold[:int(self._fold_size/2),:] = labels_0[offset:offset+int(self._fold_size/2),:]
                label_fold[int(self._fold_size/2):,:] = labels_1[offset:offset+int(self._fold_size/2),:]

                self._image_folds.append(np.array(image_fold))
                self._label_folds.append(np.array(label_fold))

                # --- Shuffle ---
                perm = np.arange(2*int(self._fold_size/2))
                np.random.shuffle(perm)
                self._image_folds[i] = self._image_folds[i][perm]
                self._label_folds[i] = self._label_folds[i][perm]

                offset += int(self._fold_size/2)

            image_fold = []
            image_fold += list(images_0[offset:])
            image_fold += list(images_1[offset:])

            label_fold = np.zeros((2*(len(images_0)-offset), 2))
            label_fold[:len(images_0)-offset,:] = labels_0[offset:]
            label_fold[len(images_0)-offset:,:] = labels_1[offset:]

            self._image_folds.append(np.array(image_fold))
            self._label_folds.append(np.array(label_fold))

            # --- Shuffle ---
            perm = np.arange(len(self._image_folds[cross_validation_folds-1]))
            np.random.shuffle(perm)
            self._image_folds[cross_validation_folds-1] = self._image_folds[cross_validation_folds-1][perm]
            self._label_folds[cross_validation_folds-1] = self._label_folds[cross_validation_folds-1][perm]

            imageset = []
            labelset = []
            for i, fold in enumerate(self._image_folds):
                if i != self._current_fold:
                    imageset += list(fold)
            for i, fold in enumerate(self._label_folds):
                if i != self._current_fold:
                    labelset += list(fold)
            self._Training = SubSet(np.array(imageset), np.array(labelset))				
            self._Validation = SubSet(np.array(self._image_folds[self._current_fold]),
                                    np.array(self._label_folds[self._current_fold]))
            print('Creating folds...done.')

            if normalize:
                print('Computing mean...')
#				mean, std = online_std(self._Training.images)
                mean = online_flattened_mean(self._Training.images)
                print('Computing mean...done.')

                self.Normalization = True
                self._Training.Normalization = True
                self._Validation.Normalization = True
                self._Training.setNormalizationParameters(mean)
                self._Validation.setNormalizationParameters(mean)
            else:
                self.Normalization = False
                self._Training.Normalization = False
                self._Validation.Normalization = False


        print('Init Dataset...done.')

    def next_fold(self):
        self._current_fold += 1

        imageset = []
        labelset = []
        for i, fold in enumerate(self._image_folds):
            if i != self._current_fold:
                imageset += list(fold)
        for i, fold in enumerate(self._label_folds):
            if i != self._current_fold:
                labelset += list(fold)
        self._Training = SubSet(np.array(imageset), np.array(labelset))	
        self._Validation = SubSet(np.array(self._image_folds[self._current_fold]),
                                np.array(self._label_folds[self._current_fold]))

        if self.Normalization:
            print('Computing mean and std image...')
#			mean, std = online_std(self._Training.images)
            mean = online_flattened_mean(self._Training.images)
            print('Computing mean and std image...done.')

            self._Training.Normalization = True
            self._Validation.Normalization = True
            self._Training.setNormalizationParameters(mean)
            self._Validation.setNormalizationParameters(mean)
        else:
            self._Training.Normalization = False
            self._Validation.Normalization = False

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

    def setNormalizationParameters(self, mean):
        self._mean = mean

    def next_batch(self, batch_size, bases3d=True):
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

#        image_batch = np.array([ self.getImageArray(image_path) for image_path in self._images[start:end]])
        label_batch = self._labels[start:end]
        image_batch = np.array([ np.zeros((30,512,512)) if t[0]==0 else np.ones((30,512,512)) for t in label_batch])
        
        image_batch = np.swapaxes(image_batch, 1, 2)
        image_batch = np.swapaxes(image_batch, 2, 3)

        # --- Only in case of 3D model ---
        if bases3d:
            image_batch = np.expand_dims(image_batch, axis=4)			

        return image_batch, label_batch


    def getImageArray(self, image_path):
        """
        Returns:
            Numpy array of the loaded image
        Args:
            image_path: Path of image to read from.
        """
        if self.Normalization:
            return normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(image_path))) - self._mean
        else:
            return normalize_image(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
