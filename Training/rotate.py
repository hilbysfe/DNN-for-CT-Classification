import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage.interpolation import rotate

def rotate_images(dataset_path, n):	
	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + dataset_path + '\\Training'

	filenames = [filename for filename in os.listdir(image_dir + '\\') if filename.endswith(".mhd")]
	files = [sitk.ReadImage(image_dir + '\\' + filename) for filename in filenames]
	images = np.array([sitk.GetArrayFromImage(file) for file in files])
	print("Loaded " + dataset_path)
	print(images.shape)

	# -90 / 90
	new_image_dir = image_dir + '_augmented' + str(n) + "\\"
	os.mkdir(new_image_dir)

	for i, image in enumerate(images):
		for j in range(1,n+1):
			angle = np.random.randint(-90,90)
			new_data = rotate(image, angle, (1,2))
			
			w = new_data.shape[1]
			h = new_data.shape[2]
			
			img = sitk.GetImageFromArray(new_data[:,int(np.floor(w/2-64)):int(np.floor(w/2+64)),int(np.floor(h/2-64)):int(np.floor(h/2+64))])
			img.SetSpacing(files[i].GetSpacing())
			
			filename = filenames[i].split(".mhd")[0] + '_' + str(j) + '.mhd'

			sitk.WriteImage(img, new_image_dir + filename)
	print("-90/90 done. ")
	
	# 0 / 360
	new_image_dir = image_dir + '_augmented' + str(n) + "-360\\"
	os.mkdir(new_image_dir)

	for i, image in enumerate(images):
		for j in range(1,n+1):
			angle = np.random.randint(360)
			new_data = rotate(image, angle, (1,2))
			
			w = new_data.shape[1]
			h = new_data.shape[2]
			
			img = sitk.GetImageFromArray(new_data[:,int(np.floor(w/2-64)):int(np.floor(w/2+64)),int(np.floor(h/2-64)):int(np.floor(h/2+64))])
			img.SetSpacing(files[i].GetSpacing())
			
			filename = filenames[i].split(".mhd")[0] + '_' + str(j) + '.mhd'

			sitk.WriteImage(img, new_image_dir + filename)
	print("0/360 done. ")

rotate_images('Normalized_Resampled_128x128x30', 3)
rotate_images('Normalized_Resampled_128x128x30', 5)
rotate_images('Normalized_RigidAligned_128x128x30', 3)
rotate_images('Normalized_RigidAligned_128x128x30', 5)		
rotate_images('Normalized_Skullstripped_128x128x22', 3)
rotate_images('Normalized_Skullstripped_128x128x22', 5)		
		