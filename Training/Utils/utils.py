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

    # --- Filter out the ones missing from Database ---
    patients = [ name for name in patients if name in label_dict.keys() ]	

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
    
    num_examples = len(class0_images) + len(class1_images)

    # --- Schuffle both classes ---
    perm = np.arange(len(class0_images))
    np.random.shuffle(perm)
    class0_images = np.array(class0_images)[perm]
    
    perm = np.arange(len(class1_images))
    np.random.shuffle(perm)
    class1_images = np.array(class1_images)[perm]
    
    # --- Split into balanced test-set and unbalanced training ---
    test_size = int(num_examples*test_ratio)    
    test_points = dict(
        zip(np.concatenate((class0_images[:int(test_size/2)], class1_images[:int(test_size/2)])),
            np.concatenate((np.zeros((int(test_size/2),), dtype=np.int), np.ones((int(test_size/2),), dtype=np.int)))))
    
    training_points = dict(
        zip(np.concatenate((class0_images[int(test_size/2):], class1_images[int(test_size/2):])),
            np.concatenate((np.zeros((len(class0_images)-int(test_size/2),), dtype=np.int), np.ones((len(class1_images)-int(test_size/2),), dtype=np.int)))))
    
    return training_points, test_points


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

    def __init__(self, training_points, test_points, cross_validation_folds=0, normalize=False):
        print('Init Dataset...')
        # === TEST-SET ===
        # --- Split and shuffle ---
        perm = np.arange(int(len(test_points)/2))
        np.random.shuffle(perm)
        test_images0 = np.array([ image for image in test_points if test_points[image] == 0])[perm]
        test_images1 = np.array([ image for image in test_points if test_points[image] == 1])[perm]
        test_labels0 = np.array([ np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(test_images0)) ])
        test_labels1 = np.array([ np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(test_images1)) ])
        self._Test = SubSet(test_images0, test_images1, test_labels0, test_labels1)
        
        # === TRAINING-SET ===
        # --- Split and shuffle ---
        training_images0 = np.array([ image for image in training_points if training_points[image] == 0])
        perm = np.arange(len(training_images0))
        np.random.shuffle(perm)
        training_images0 = training_images0[perm]

        training_images1 = np.array([ image for image in training_points if training_points[image] == 1])
        perm = np.arange(len(training_images1))
        np.random.shuffle(perm)
        training_images1 = training_images1[perm]

        # --- Balance out ---
        balanced_size = len(training_images0) if len(training_images0)<len(training_images1) else len(training_images1)
        
        training_images0 = training_images0[:balanced_size]
        training_images1 = training_images1[:balanced_size]
        training_labels0 = np.array([ np.ndarray((2,), buffer=np.array([0, 1]), dtype=int) for i in range(len(training_images0)) ])
        training_labels1 = np.array([ np.ndarray((2,), buffer=np.array([1, 0]), dtype=int) for i in range(len(training_images1)) ])
        
        if cross_validation_folds == 0:
            
            self._Training = SubSet(training_images0, training_images1, training_labels0, training_labels1)
            
            if normalize:
                self.Normalization = True
                self._Training.Normalization = True
                self._Test.Normalization = True
                
                print('Computing mean...')
                mean = online_flattened_mean(self._Training.images)
                print('Computing mean...done.')

                self._Training.setNormalizationParameters(mean)
                self._Test.setNormalizationParameters(mean)
            else:
                self.Normalization = False
                self._Training.Normalization = False
                self._Test.Normalization = False

        else:
            print('Creating folds...')
            
            # === TRAINING + VALIDATION-SET ===
            self._current_fold = 0
            self._point_per_fold = int(balanced_size/cross_validation_folds)
            
            perm = np.arange(2*self._point_per_fold)
            np.random.shuffle(perm)
            
            # --- Prepare folds ---
            self._image0_folds = []
            self._label0_folds = []
            self._image1_folds = []
            self._label1_folds = []
            offset = 0
            for i in range(cross_validation_folds):
                self._image0_folds.append(
                    training_images0[offset:offset+self._point_per_fold]
                )
                self._image1_folds.append(
                    training_images1[offset:offset+self._point_per_fold]
                )
                self._label0_folds.append(
                    training_labels0[offset:offset+self._point_per_fold]
                )
                self._label1_folds.append(
                    training_labels1[offset:offset+self._point_per_fold]
                )
                
                offset += self._point_per_fold

            # --- Init sets with 1st fold ---
            training_imgset0 = np.concatenate( [ fold for i,fold in enumerate(self._image0_folds) if i != self._current_fold] )
            training_labelset0 = np.concatenate( [ fold for i,fold in enumerate(self._label0_folds) if i != self._current_fold] )
            validation_imgset0 = self._image0_folds[self._current_fold]
            validation_labelset0 = self._label0_folds[self._current_fold]
            training_imgset1 = np.concatenate( [ fold for i,fold in enumerate(self._image1_folds) if i != self._current_fold] )
            training_labelset1 = np.concatenate( [ fold for i,fold in enumerate(self._label1_folds) if i != self._current_fold] )
            validation_imgset1 = self._image1_folds[self._current_fold]
            validation_labelset1 = self._label1_folds[self._current_fold]
            
            self._Training = SubSet(training_imgset0, training_imgset1, training_labelset0, training_labelset1)
            self._Validation = SubSet(validation_imgset0, validation_imgset1, validation_labelset0, validation_labelset1)
            print('Creating folds...done.')

            if normalize:
                print('Computing mean...')
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
        
        training_imgset0 = np.concatenate( [ fold for i,fold in enumerate(self._image0_folds) if i != self._current_fold] )
        training_labelset0 = np.concatenate( [ fold for i,fold in enumerate(self._label0_folds) if i != self._current_fold] )
        validation_imgset0 = self._image0_folds[self._current_fold]
        validation_labelset0 = self._label0_folds[self._current_fold]
        training_imgset1 = np.concatenate( [ fold for i,fold in enumerate(self._image1_folds) if i != self._current_fold] )
        training_labelset1 = np.concatenate( [ fold for i,fold in enumerate(self._label1_folds) if i != self._current_fold] )
        validation_imgset1 = self._image1_folds[self._current_fold]
        validation_labelset1 = self._label1_folds[self._current_fold]

        self._Training = SubSet(training_imgset0, training_imgset1, training_labelset0, training_labelset1)
        self._Validation = SubSet(validation_imgset0, validation_imgset1, validation_labelset0, validation_labelset1)


        if self.Normalization:
            print('Computing mean and std image...')
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
        self._images = np.concatenate( (images0, images1) )
        self._labels = np.concatenate( (labels0, labels1) )
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
    
    def setNormalizationParameters(self, mean):
        self._mean = mean

    def next_batch(self, batch_size, bases3d=False):
        """
        Return the next `batch_size` examples from this data set.
        Args:
            batch_size: Batch size.
        """
        assert batch_size <= 2*self._num_examples
        
        start = self._index_in_epoch
        self._index_in_epoch += int(batch_size/2)
        if self._index_in_epoch >= self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images0 = self._images0[perm]
            self._labels0 = self._labels0[perm]
            self._images1 = self._images1[perm]
            self._labels1 = self._labels1[perm]

            start = 0
            self._index_in_epoch = batch_size


        end = self._index_in_epoch

        perm = np.arange(batch_size)
        np.random.shuffle(perm)
        
#        image_batch = np.array([ self.getImageArray(image_path) for image_path in self._images[start:end]])
        label_batch = np.concatenate( (self._labels0[start:end], self._labels1[start:end]) )[perm]
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
