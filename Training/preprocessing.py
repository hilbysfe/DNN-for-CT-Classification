import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
import shutil
from multiprocessing import Pool
from itertools import repeat

from Preprocessing import preprocessing_utils

WINDOW = 10
OVERLAP = 3

#NCCT_MRCLEAN = 			r'E:\MRCLEAN\NCCT_THICK'
#NCCT_SKULL_MRCLEAN = 		r'E:\MRCLEAN\NCCT_THICK_SKULLSTRIPPED'
#MIP_MRCLEAN = 			r'E:\MRCLEAN\CTA_MIP'
#MIP_SKULL_MRCLEAN = 		r'E:\MRCLEAN\CTA_SKULLSTRIPPED_MIP'

#NCCT_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK'
#NCCT_REGISTRY_RESAMPLED = 	r'E:\REGISTRY\NCCT_THICK_RESAMPLED\'
#NCCT_SKULL_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED'
#NCCT_SKULL_REGISTRY_RESAMPLED =	r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
#MIP_REGISTRY = 			r'E:\REGISTRY\CTA_MIP'
#MIP_SKULL_REGISTRY = 		r'E:\REGISTRY\CTA_SKULLSTRIPPED_MIP'

NCCT_MRCLEAN = 			r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK'
NCCT_SKULL_MRCLEAN = 		r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK_SKULLSTRIPPED'
MIP_MRCLEAN = 			r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_MIP'
MIP_SKULL_MRCLEAN = 		r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_SKULLSTRIPPED_MIP'

NCCT_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK'
NCCT_REGISTRY_RESAMPLED = 	r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_RESAMPLED'
NCCT_SKULL_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED'
NCCT_SKULL_REGISTRY_RESAMPLED =	r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
MIP_REGISTRY = 			r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_MIP'
MIP_SKULL_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_SKULLSTRIPPED_MIP'

CTA_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_THIN'
CTA_REGISTRY_RESAMPLED = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_THIN_RESAMPLED'

NCCT = r'C:\Users\Adam\Registry\NCCT_BL\all'
NCCT_THIN = r'C:\Users\Adam\Registry\NCCT_BL\ST_THIN'

def soft_tissue(file):
	try:
		kernel = preprocessing_utils.GetConvKernel(file)
	except:
		kernel = ""

	try:
		series = preprocessing_utils.GetSeriesDescription(file)
	except:
		series = ""

	return	kernel == 'H31s' or \
			kernel == 'h31s' or \
			'h31s' in series or \
			'H31s' in series or \
			( preprocessing_utils.GetWindowCenter(file) < 100.0 and preprocessing_utils.GetWindowWidth(file) < 200.0 )


def softtissue_thin_slice_selection(patient, threshold):
	try:
		if os.path.exists(os.path.join(NCCT_THIN, patient)):
			print(patient + ' done.')
			return

		# Collect all folder below patient folder which contains DICOM
		folder_dict = preprocessing_utils.CollectDICOMFolders(os.path.join(NCCT, patient))
		blacklist = []
		sliceMap = {}
		for folder in folder_dict:
			dicomFiles = folder_dict[folder]
			if len(dicomFiles) == 0:
				blacklist.append(folder)
				continue
			# Sort files
			sortedFiles = preprocessing_utils.sort_files(dicomFiles, map=preprocessing_utils.GetZLocation)
			# Pick a slice
			if len(sortedFiles) < 11:
				blacklist.append(folder)
				continue
			slice = sortedFiles[10]
			# Check soft-tissue criteria
			if not soft_tissue(slice):
				blacklist.append(folder)
				continue
			else:
				# Check slice thickness criteria
				sliceList = [round(preprocessing_utils.GetSliceThickness(file), 1) for file in sortedFiles]
				if len(set(sliceList)) != 1:
					print(folder + ': multiple slice thickness.')
					blacklist.append(folder)
					continue
				else:
					thickness = sliceList[0]
#					if thickness > threshold:
#						blacklist.append(folder)
#						continue
#					else:
					sliceMap[folder] = thickness

		# Remove blacklisted folders
		for f in blacklist:
			del folder_dict[f]

		# Remove Coronal / Saggital series
#		folder_dict = {folder: folder_dict[folder] for folder in folder_dict if not preprocessing_utils.Sag_or_Cor(folder_dict[folder])}

		# Delete localizers
		for folder in folder_dict:
			localizers = preprocessing_utils.DetectLocalizers(folder_dict[folder])
			for file in localizers:
				folder_dict[folder].remove(file)
	
		# Check if any series left
		if len(folder_dict) == 0:
			print(patient + ': no decent (baseline) NCCT.')
			return

		# Select baseline series
		baselines = preprocessing_utils.SelectBaseline(folder_dict)	
	
		# Copy final series
		for folder in baselines:
			preprocessing_utils.CopyFiles(baselines[folder], os.path.join(NCCT_THIN, patient, str(sliceMap[folder])))
		
		print(patient + ' done.')
	except Exception as e:
		print(patient + ' failed: %s' %e)


if __name__ == '__main__':
#	patients = os.listdir(NCCT)
	softtissue_thin_slice_selection('R1251', 2.0)
#	with Pool() as p:
#		p.starmap(softtissue_thin_slice_selection, zip(patients, repeat(2.0)))

