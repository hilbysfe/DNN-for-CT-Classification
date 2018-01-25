import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
import shutil
# from multiprocessing import Pool
# import scipy.ndimage
import sys

def GetZLocation(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0020|0032').split('\\')[2])

def sort_files(files, map):
	return sorted(files, key=map)


def MIP(input_image, window, overlap):
	# if not os.path.exists(os.path.join(rootTarget, patient)):
	# try:

	spacing = input_image.GetSpacing()
	input_data = sitk.GetArrayFromImage(input_image)

	# Compute MIP
	mip_slices = []
	offset = 0
	while offset < input_data.shape[0]:
		chunk = input_data[offset:min(offset+window, input_data.shape[0]), :, :]
		new_slice = np.max(chunk, axis=0)
		mip_slices.append(new_slice)
		offset += (window - overlap)

	# Create image
	mip_image = sitk.GetImageFromArray(np.array(mip_slices)[:, ::-1, ::-1])
	mip_image.SetSpacing((spacing[0], spacing[1], spacing[2]*float(window - overlap)))

	# os.makedirs(os.path.join(rootTarget, patient))
	# sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))

#	os.makedirs(os.path.join(rootTarget, "mip"))
#	sitk.WriteImage(mip_image, os.path.join(rootTarget, "mip", patient + '.mha'))

	return mip_image

def MIP_2D(input_image, threshold):
	# if not os.path.exists(os.path.join(rootTarget, patient)):
	# try:

	input_data = sitk.GetArrayFromImage(input_image)

	input_data[input_data > 400] = 0

	# Delete high intensity rim around brain
	max_z = np.max(np.nonzero(input_data)[0])
	min_z = np.min(np.nonzero(input_data)[0])

	for z in range(min_z, max_z+1):
		max_y = np.max(np.nonzero(input_data[z,:,:])[0])
		min_y = np.min(np.nonzero(input_data[z,:,:])[0])
		for y in range(min_y, max_y+1):
			if len(np.nonzero(input_data[z,y,:])[0]) == 0:
				continue
			max_x = np.max(np.nonzero(input_data[z,y,:])[0])
			min_x = np.min(np.nonzero(input_data[z,y,:])[0])
			for x in range(min_x, max_x+1):
				if input_data[z,y,x] > 70 and \
					(input_data[z,y,x+1] == 0 or input_data[z,y,x-1] == 0
					 or input_data[z,y+1,x] == 0 or input_data[z,y-1,x] == 0):
					input_data[z,y,x] = 0

	# Compute MIP
	new_slice = np.max(input_data, axis=0)

	# Create image
	mip_image = sitk.GetImageFromArray(np.array(new_slice))

	# os.makedirs(os.path.join(rootTarget, patient))
	# sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))

#	os.makedirs(os.path.join(rootTarget, "mip"))
#	sitk.WriteImage(mip_image, os.path.join(rootTarget, "mip", patient + '.mha'))

	return mip_image

def MIP_DICOM(patient):

	# if not os.path.exists(os.path.join(rootTarget, patient)):
	# try:
	# Load image
	DicomFolder = os.path.join(rootDicom)
	reader = sitk.ImageSeriesReader()
	series_found = reader.GetGDCMSeriesIDs(DicomFolder)
	if len(series_found) != 1:
		print(patient + ' more series found.')
	filenames = reader.GetGDCMSeriesFileNames(DicomFolder, series_found[0])
	reader.SetFileNames(filenames)
	input_image = reader.Execute()
	spacing = input_image.GetSpacing()
	input_data = sitk.GetArrayFromImage(input_image)
	sorted_files = sort_files(filenames, map=GetZLocation)

	# Retrieve location information
	sliceLocation = [ GetZLocation(file) for file in sorted_files ]

	# Compute MIP
	mip_slices = []
	i = 0
	offset = 0
	while i < len(sliceLocation):
		gathered = 0.0
		collectedSlices = []
		start = i
		i = offset

		# Collect slice regarding location
		while gathered < WINDOW and i < len(sliceLocation):
			collectedSlices.append(i)
			gathered += abs(sliceLocation[i] - sliceLocation[max(i-1, 0)])
			if gathered < WINDOW - OVERLAP:
				offset = i
			i += 1
		new_slice = np.max(input_data[start:i, :, :], axis=0)
		mip_slices.append(new_slice)

	# Create image
	mip_image = sitk.GetImageFromArray(np.array(mip_slices))
	mip_image.SetSpacing(spacing)
	
	# os.makedirs(os.path.join(rootTarget, patient))
	# sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))
	
	os.makedirs(os.path.join(rootTarget, "mip"))
	sitk.WriteImage(mip_image, os.path.join(rootTarget, "mip", patient + '.mha'))

		# print(patient + ' done.')
	# except:
		# print(sys.exc_info()[0])


def MIP_MHA(patient):
	if not os.path.exists(os.path.join(rootTarget, patient)):
		try:
			# Load image
			ImagePath = os.path.join(rootSource, patient, patient + '.mha')
			DicomFolder = os.path.join(rootDicom, patient)
			reader = sitk.ImageSeriesReader()
			series_found = reader.GetGDCMSeriesIDs(DicomFolder)
			if len(series_found) != 1:
				print(patient + ' more series found.')
			filenames = reader.GetGDCMSeriesFileNames(DicomFolder, series_found[0])
			sorted_files = sort_files(filenames, map=GetZLocation)

			input_image = sitk.ReadImage(ImagePath)
			input_data = sitk.GetArrayFromImage(input_image)

			# Retrieve location information
			sliceLocation = [ GetZLocation(file) for file in sorted_files ]

			# Compute MIP
			mip_slices = []
			i = 0
			offset = 0
			while i < len(sliceLocation):
				gathered = 0.0
				collectedSlices = []
				start = i
				i = offset

				# Collect slice regarding location
				while gathered < WINDOW and i < len(sliceLocation):
					collectedSlices.append(i)
					gathered += abs(sliceLocation[i] - sliceLocation[max(i-1, 0)])
					if gathered < WINDOW - OVERLAP:
						offset = i
					i += 1

				new_slice = np.max(input_data[start:i, :, :], axis=0)
				mip_slices.append(new_slice)

			# Create image
			mip_image = sitk.GetImageFromArray(np.array(mip_slices))
			mip_image.SetSpacing(spacing)
			
			os.makedirs(os.path.join(rootTarget, patient))
			sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))

			print(patient + ' done.')
		except:
			print(patient + ' failed.')

WINDOW = 15
OVERLAP = 3

rootDicom = r"D:\Adam Hilbert\Data\MRCLEAN_REGISTRY\CTA_BL\R0008\14529377\1.3.6.1.4.1.40744.9.297226749760607436395914976335537367268"
rootTarget = r"D:\Adam Hilbert\Data\MRCLEAN_REGISTRY\CTA_BL\R0008"
			
if __name__ == '__main__':

	patients = "R0008"

	MIP_DICOM(patients)
	
	# with Pool() as p:
		# p.starmap(MIP_DICOM, patients)