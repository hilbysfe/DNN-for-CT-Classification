import numpy as np
import os
import SimpleITK as sitk
from shutil import copyfile
import openpyxl as ox
# import matplotlib.pyplot as plt
# %pylab inline

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def normalize_images(dataset_path):

	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + dataset_path
	
	training_images, validation_images, test_images = read_dataset(image_dir)
	# print(training_images)
	# Spread images
	os.mkdir(image_dir + '\\Training\\')
	for image in training_images:		
		sitk.WriteImage(sitk.ReadImage(image), image_dir + '\\Training\\' + image.split("\\")[-1])
		
	os.mkdir(image_dir + '\\Validation\\')
	for image in validation_images:
		sitk.WriteImage(sitk.ReadImage(image), image_dir + '\\Validation\\' + image.split("\\")[-1])
	
	os.mkdir(image_dir + '\\Test\\')		
	for image in test_images:
		sitk.WriteImage(sitk.ReadImage(image), image_dir + '\\Test\\' + image.split("\\")[-1])
	
	image_dir = root + dataset_path + '\\Training\\'
	filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".mhd")]
	files = [sitk.ReadImage(image_dir + filename) for filename in filenames]
	images = np.array([sitk.GetArrayFromImage(file) for file in files])
			
	mean_data = np.mean(images, axis=0)
	sitk.WriteImage(sitk.GetImageFromArray(mean_data), root + dataset_path + '\\Mean.mhd')
	std_data = np.std(images, axis=0)
	sitk.WriteImage(sitk.GetImageFromArray(std_data), root + dataset_path + '\\Std.mhd')
						
	# --- Save image ---
	new_image_dir = root + 'Normalized_' + dataset_path + '\\'
	os.mkdir(new_image_dir)
	new_image_dir = root + 'Normalized_' + dataset_path + '\\Training\\'
	os.mkdir(new_image_dir)
	for i, image in enumerate(images):
		img_new = sitk.GetImageFromArray(div0(image - mean_data, std_data))
		img_new.SetSpacing(files[i].GetSpacing())
		filename = filenames[i]

		sitk.WriteImage(img_new, new_image_dir + filename)
		
	# Validation
	image_dir = root + dataset_path + '\\Validation\\'
	filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".mhd")]
	files = [sitk.ReadImage(image_dir + filename) for filename in filenames]
	images = np.array([sitk.GetArrayFromImage(file) for file in files])
	# --- Save image ---
	new_image_dir = root + 'Normalized_' + dataset_path + '\\Validation\\'
	os.mkdir(new_image_dir)
	for i, image in enumerate(images):
		img_new = sitk.GetImageFromArray(div0(image - mean_data, std_data))
		img_new.SetSpacing(files[i].GetSpacing())
		filename = filenames[i]

		sitk.WriteImage(img_new, new_image_dir + filename)
	
	# Test
	image_dir = root + dataset_path + '\\Test\\'
	filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".mhd")]
	files = [sitk.ReadImage(image_dir + filename) for filename in filenames]
	images = np.array([sitk.GetArrayFromImage(file) for file in files])
	# --- Save image ---
	new_image_dir = root + 'Normalized_' + dataset_path + '\\Test\\'
	os.mkdir(new_image_dir)
	for i, image in enumerate(images):
		img_new = sitk.GetImageFromArray(div0(image - mean_data, std_data))
		img_new.SetSpacing(files[i].GetSpacing())
		filename = filenames[i]

		sitk.WriteImage(img_new, new_image_dir + filename)
		

	
	
	
	
	