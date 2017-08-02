import numpy as np
import os
import SimpleITK as sitk
from shutil import copyfile
import openpyxl as ox
# import matplotlib.pyplot as plt
# %pylab inline

def read_dataset(image_dir, validation_ratio=0.25, test_ratio=0.15):
	"""
	Function to read up images and labels.
	Store only paths as images would fit to memory.

	MRS@90 hardcoded as label attribute in label_filename -> column 'AE'
	"""
	
	label_filename = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\MRCLEAN\\MRCLEAN_MRSDICH.xlsx'
	
	followid_attribute = 'AB2:AB501'
	label_attribute = 'AE2:AE501'

	# --- Retrieve all patients we have images from ---
	patients = [name for name in os.listdir(image_dir) if name.split('.', 1)[1]=="mhd"]

	all_samples = len(patients)

	# --- Load labels from file ---
	labels_wb = ox.load_workbook(label_filename)
	labels_ws = labels_wb.active

	label_dict = {key[0].value.zfill(4): value[0].value 
		for (key, value) in zip(labels_ws[followid_attribute], labels_ws[label_attribute]) }

	# --- Store images(optionally more per patient) + labels ---
	all_images = [ image_dir + '\\' + name for name in patients ]
	all_labels = [ label_dict[name.split('.',1)[0].split('_')[0]] for name in patients ]				

	all_labels = np.array( all_labels )
	all_images = np.array( all_images )

	uni, classes = np.unique(all_labels, return_counts = True)
	num_classes = np.shape(classes)[0]

	num_examples = np.min(classes) * num_classes
	smallest_class = dict(zip(classes, uni))[np.min(classes)]

	indices = []
	for i in range(num_classes):
		indices_class_i = np.where(all_labels == i)[0]
		if i != smallest_class:
			perm = np.arange(min(classes))
			np.random.shuffle(perm)
			indices_class_i = indices_class_i[perm]

		indices = np.append(indices, indices_class_i)

	indices = [int(x) for x in indices]

	all_labels = all_labels[indices]
	all_images = all_images[indices]


	# Select original images for validation/test and augmented for training
	original = np.array([image.split("\\")[-1].split('.')[0] for image in all_images])
	perm = np.arange(np.shape(original)[0])
	np.random.shuffle(perm)
	original = original[perm]

	# Calculate and cut the subsets
	validation_size = int(np.shape(original)[0]*validation_ratio)
	test_size = int(np.shape(original)[0]*test_ratio)

	# ONLY IN CASE OF CLASSES = 2
	validation_patIDs = []
	i = 0
	while len(validation_patIDs) < validation_size:
		pat = original[i]
		if len(validation_patIDs) < validation_size/2 and label_dict[pat] == 0:
			validation_patIDs.append(pat)
		if len(validation_patIDs) >= validation_size/2 and label_dict[pat] == 1:
			validation_patIDs.append(pat)
		i += 1

	test_patIDs = []
	i=0
	while len(test_patIDs) < test_size:
		pat = original[i]
		if not pat in validation_patIDs:
			if len(test_patIDs) < test_size/2 and label_dict[pat] == 0:
				test_patIDs.append(pat)
			if len(test_patIDs) >= test_size/2 and label_dict[pat] == 1:
				test_patIDs.append(pat)
		i += 1

	training_patIDs = [pat for pat in original if not pat in validation_patIDs and not pat in test_patIDs]
	
	training_images = []
	training_labels = []
	validation_images = []
	validation_labels = []	
	test_images = []
	test_labels = []
		
	for image in all_images:
		pat = image.split("\\")[-1].split('.')[0].split('_')[0]
		if pat in training_patIDs:
			training_images.append(image)
			training_labels.append(label_dict[pat])

	for image in all_images:
		pat = image.split("\\")[-1].split('.')[0]
		if not "_2" in pat and pat.split('_')[0] in validation_patIDs:
			validation_images.append(image)
			validation_labels.append(label_dict[pat.split('_')[0]])

	for image in all_images:
		pat = image.split("\\")[-1].split('.')[0]
		if not "_2" in pat and pat.split('_')[0] in test_patIDs:
			test_images.append(image)
			test_labels.append(label_dict[pat.split('_')[0]])

	
	return training_images, validation_images, test_images




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
		
normalize_images('Resampled_256x256x30')
	
	
	
	
	