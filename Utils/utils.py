from __future__ import division

import os
import SimpleITK as sitk
import numpy as np
import openpyxl as ox

def read_datasets(data_dir, images_root_sub, image_sub, label_root_sub, label_filename, validation_ratio):
    """
    Function to read up images and labels.
    Store only paths as images would fit to memory.

    MRS@90 hardcoded as label attribute in label_filename -> column 'AE'
    """
    
    followid_attribute = 'AB2:AB501'
    label_attribute = 'AE2:AE501'
    
    # --- Retrieve all patients we have images from ---
    patients = [name for name in os.listdir(data_dir + images_root_sub)
            if os.path.isdir(os.path.join(data_dir + images_root_sub, name))]
    # image_sub = "CT24h\\thick\\"
    all_samples = len(patients)

    # --- Store image paths ---
    all_images = [data_dir + images_root_sub + name.zfill(4) + '\\' + image_sub + 'pat' + name.zfill(4) + '.mhd' \
                    for name in patients ]
    
    # --- Load labels from file ---
    labels_wb = ox.load_workbook(data_dir + label_root_sub + label_filename)
    labels_ws = labels_wb.active

    label_dict = {key[0].value.zfill(4): value[0].value 
                  for (key, value) in zip(labels_ws[followid_attribute], labels_ws[label_attribute]) }

    labels = np.array( [ label_dict[name] for name in patients] )

    # --- Shuffle example to split fairly ---	
    perm = np.arange(all_samples)
    np.random.shuffle(perm)

    # rounding instead?
    validation_size = int(all_samples*validation_ratio)
    
    # --- Split data to training and validation set ---
    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []

    for i in range(all_samples):
        if i < validation_size:
            validation_images.append(all_images[perm[i]])
            validation_labels.append(labels[perm[i]])
        else:
            training_images.append(all_images[perm[i]])
            training_labels.append(labels[perm[i]])

    # --- Return 2 DataSet objects for training and validation set ---	
    return DataSet(np.array(training_images), np.array(training_labels)), \
            DataSet(np.array(validation_images), np.array(validation_labels))



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
    def images(self):
        return self._images

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
