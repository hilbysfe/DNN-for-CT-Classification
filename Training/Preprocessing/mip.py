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