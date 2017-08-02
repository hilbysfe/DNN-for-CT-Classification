import numpy as np
import os
import SimpleITK as sitk



def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c
	
def normalize(dataset):
	""" 
	Returns:
		the normalized datapoints with 0 mean and unit variance of all features across the dataset. 
	Input:
		dataset - [N, x1, ... xm] shaped array, where N is the size of the dataset and m is the number of features per datapoint.
	"""
	
	# --- Compute mean / std ---	
	mean = np.zeros(np.shape(dataset[0]))
	
	mean_data = np.mean(images, axis=0)
	sitk.WriteImage(sitk.GetImageFromArray(mean_data), root + dataset_path + '\\Mean.mhd')
	std_data = np.std(images, axis=0)
	sitk.WriteImage(sitk.GetImageFromArray(std_data), root + dataset_path + '\\Std.mhd')
						
	# --- Normalize ---
	for i, image in enumerate(images):
		img_new = sitk.GetImageFromArray(div0(image - mean_data, std_data))
		img_new.SetSpacing(files[i].GetSpacing())
		filename = filenames[i]

		sitk.WriteImage(img_new, new_image_dir + filename)
	
	
	
	
	