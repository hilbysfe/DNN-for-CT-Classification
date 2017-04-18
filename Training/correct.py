import numpy as np
import SimpleITK as sitk
import os

root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\Original_512x512x30'

filenames = [filename for filename in os.listdir(root + '_old\\') if filename.endswith(".mhd")]

for filename in filenames:
	image = sitk.GetArrayFromImage(sitk.ReadImage(root + '_old\\' + filename))
	image[image < -1000] = -1000
	sitk.WriteImage(sitk.GetImageFromArray(image), root + '\\' + filename)
