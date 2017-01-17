"""
Computes the skull stripping of all registered data in the dataset.

Here we ommit slices before skullstripping which doesn't carry significant information for the task and
keep only slices between 5-27. This reserves a little slack for future experiments.
"""

import skullStripping2D as ss
import os
import SimpleITK as sitk
import numpy as np

data_dir = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\'
images_root_sub = 'MRCLEAN_CT24h\\'
image_sub = '\CT24h\\thick\\alignedRigid\\'

# --- Retrieve all patients we have images from ---
patients = [name.zfill(4) for name in os.listdir(data_dir + images_root_sub)
                if os.path.isdir(os.path.join(data_dir + images_root_sub, name))]

for patient in patients:
	filename = 'result.mhd'
	fullpath = data_dir + images_root_sub + patient + image_sub + filename

	# --- Read Image ---
	image = sitk.ReadImage(fullpath)

	# --- Load Data ---
	img_data = sitk.GetArrayFromImage(image)

	# --- Skullstripping ---
	data_new = ss.skullstripping(img_data[5:27,:,:])

	# --- Save image ---
	img_new = sitk.GetImageFromArray(data_new)
	newFileName = 'ct24h_registered_skullstripped.mhd'

	sitk.WriteImage(img_new, data_dir + images_root_sub + patient + image_sub + newFileName)

	print(patient + ' done.')
    