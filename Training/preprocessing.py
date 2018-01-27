import numpy as np
import SimpleITK as sitk
import os
import time
import shutil
from multiprocessing import Pool
from itertools import repeat

from Preprocessing import mip
from Preprocessing import preprocessing_utils
from Preprocessing.Registration import align
from Preprocessing.Registration import brain_segmentation as bs

WINDOW = 10
OVERLAP = 3

# NCCT_MRCLEAN = 			r'E:\MRCLEAN\NCCT_THICK'
# NCCT_SKULL_MRCLEAN = 		r'E:\MRCLEAN\NCCT_THICK_SKULLSTRIPPED'
# MIP_MRCLEAN = 			r'E:\MRCLEAN\CTA_MIP'
# MIP_SKULL_MRCLEAN = 		r'E:\MRCLEAN\CTA_SKULLSTRIPPED_MIP'

# NCCT_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK'
# NCCT_REGISTRY_RESAMPLED = 	r'E:\REGISTRY\NCCT_THICK_RESAMPLED\'
# NCCT_SKULL_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED'
# NCCT_SKULL_REGISTRY_RESAMPLED =	r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
# MIP_REGISTRY = 			r'E:\REGISTRY\CTA_MIP'
# MIP_SKULL_REGISTRY = 		r'E:\REGISTRY\CTA_SKULLSTRIPPED_MIP'

NCCT_MRCLEAN = r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK'
NCCT_SKULL_MRCLEAN = r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK_SKULLSTRIPPED'
MIP_MRCLEAN = r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_MIP'
MIP_SKULL_MRCLEAN = r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_SKULLSTRIPPED_MIP'

NCCT_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK'
NCCT_REGISTRY_RESAMPLED = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_RESAMPLED'
NCCT_SKULL_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED'
NCCT_SKULL_REGISTRY_RESAMPLED = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
MIP_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_MIP'
MIP_SKULL_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_SKULLSTRIPPED_MIP'

CTA_REGISTRY = r'E:\MRCLEAN_REGISTRY\CTA_BL'
CTA_ST = r'E:\MRCLEAN_REGISTRY_BL\CTA\CTA_ST'
CTA_THINNEST = r'E:\MRCLEAN_REGISTRY_BL\CTA\ST_THINNEST'
CTA_THINNEST_RESIZED = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_THINNEST_RESIZED'
CTA_MIP = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_MIP'
CTA_MIP_RESIZED = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_MIP_RESIZED'
CTA_2D_MIP_RESIZED = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_2D_MIP_RESIZED'
CTA_SKULLSTRIPPED = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_MIP_SKULLSTRIPPED'
CTA_SKULLSTRIPPED_RESIZED = r'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_MIP_SKULLSTRIPPED_RESIZED2'

NCCT = r'C:\Users\Adam\Registry\NCCT_BL\all'
NCCT_ST = r'E:\MRCLEAN_REGISTRY\NCCT_BL_ST'

TEST_REG = r'D:\Adam Hilbert\Data\Testing_Registration\original'
TEST_REG_OUT = r'D:\Adam Hilbert\Data\Testing_Registration\output'

NCCT_VAL = r'D:\Adam Hilbert\Data\ASPECTS_TestData\ValidationSetThick_Registry'
NCCT_THINNEST = r'D:\Adam Hilbert\Data\Registry\NCCT_BL\ST_THINNEST'
NCCT_THINNEST_RESIZED = r'D:\NCCT\ST_THINNEST_RESIZED_INT'
NCCT_THIN = r'D:\Adam Hilbert\Data\Registry\NCCT_BL\ST_THIN'
NCCT_THIN_RESIZED = r'D:\Adam Hilbert\Data\Registry\NCCT_BL\ST_THIN_RESIZED_INT'


def soft_tissue(file):
	try:
		kernel = preprocessing_utils.GetConvKernel(file)
	except:
		kernel = ""

	try:
		series = preprocessing_utils.GetSeriesDescription(file)
	except:
		series = ""

	return not 'BONEPLUS' in kernel or \
		   'h31s' in kernel or \
		   'H31s' in kernel or \
		   'J40s' in kernel or \
		   'B30f' in kernel or \
		   'b30f' in kernel or \
		   'h31s' in series or \
		   'H31s' in series or \
		   'J40s' in series or \
		   'B30f' in series or \
		   'b30f' in series or \
		   (preprocessing_utils.GetWindowCenter(file) < 300.0 and preprocessing_utils.GetWindowWidth(file) < 1000.0)

def select_slice(patient, threshold, rootSource, rootTarget):
	# --- Thick slice
	# thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootSource, patient)) if float(thickness) > threshold]
	# --- Thin slice
	# thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootSource, patient)) if thickness <= threshold]
	# --- Thinnest slice
	thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootSource, patient))]

	if len(thicknesses) == 0:
		print(patient + " no thick slice.")
		return

	min_thickness = min(thicknesses)

	# Collect all folder below patient folder which contains DICOM
	folder_dict = preprocessing_utils.CollectDICOMFolders(
		os.path.join(os.path.join(rootSource, patient), str(min_thickness)))

	if len(folder_dict.keys()) != 1:
		print(patient + ' more than 1 min thickness series.')
		return

	for f in folder_dict:
		reader = sitk.ImageSeriesReader()
		series_found = reader.GetGDCMSeriesIDs(f)
		if len(series_found) != 1:
			print(patient + ' more series found.')
			return
		filenames = reader.GetGDCMSeriesFileNames(f, series_found[0])
		reader.SetFileNames(filenames)
		input_image = reader.Execute()

		# Write image
		if not os._exists(os.path.join(rootTarget, patient)):
			os.makedirs(os.path.join(rootTarget, patient))
		sitk.WriteImage(input_image, os.path.join(rootTarget, patient, 'Scan.mha'))

		print(patient + ' done.')

def select_thick_slice(patient, rootSource, rootTarget):
	# Collect all folder below patient folder which contains DICOM
	folder_dict = preprocessing_utils.CollectDICOMFolders(os.path.join(rootSource, patient))

	if len(folder_dict.keys()) != 1:
		print(patient + ' more than 1 min thicknesses.')
		return

	for f in folder_dict:
		# Select series with more than 30 slices
		if len(folder_dict[f]) < 30:
			print(patient + ' less than 30 slices.')
			return

		reader = sitk.ImageSeriesReader()
		series_found = reader.GetGDCMSeriesIDs(f)
		if len(series_found) != 1:
			print(patient + ' more series found.')
			return
		filenames = reader.GetGDCMSeriesFileNames(f, series_found[0])
		reader.SetFileNames(filenames)
		input_image = reader.Execute()

		# Write image
		if not os._exists(os.path.join(rootTarget, patient)):
			os.makedirs(os.path.join(rootTarget, patient))
		sitk.WriteImage(input_image, os.path.join(rootTarget, patient, 'Scan.mha'))

		print(patient + ' done.')

def softtissue_selection(patient, rootSource, rootTarget):
	try:
		#		if os.path.exists(os.path.join(NCCT_THIN, patient)):
		#			print(patient + ' done.')
		#			return

		# Collect all folder below patient folder which contains DICOM
		folder_dict = preprocessing_utils.CollectDICOMFolders(os.path.join(rootSource, patient))
		blacklist = []
		sliceMap = {}
		for folder in folder_dict:
			dicomFiles = folder_dict[folder]
			if len(dicomFiles) == 0:
				blacklist.append(folder)
				continue
			# Sort files
			sortedFiles = preprocessing_utils.sort_files(dicomFiles, map=preprocessing_utils.GetZLocation)
			folder_dict[folder] = sortedFiles
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
					thickness = sliceList[10]
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
			# Select upper 20 cm
			max_loc = preprocessing_utils.GetZLocation(baselines[folder][-1])
			files_to_copy = [file for file in baselines[folder] if
							 preprocessing_utils.GetZLocation(file) > max_loc - 200]
			#			print(os.path.join(NCCT_ST, patient, str(sliceMap[folder])))
			if not os.path.exists(os.path.join(rootTarget, patient, str(sliceMap[folder]))):
				preprocessing_utils.CopyFiles(files_to_copy,
											  os.path.join(rootTarget, patient, str(sliceMap[folder])))

		print(patient + ' done.')
	except Exception as e:
		print(patient + ' failed: %s' % e)

def pipeline_NCCT(patient, rootSource, rootTarget):
	try:
		#	if os.path.exists(os.path.join(rootTarget, patient)) and len(os.listdir(os.path.join(rootTarget, patient))) == 0:
		#		shutil.rmtree(os.path.join(rootTarget, patient))
		#		return

		# Create output dict if not existing
		if not os.path.exists(os.path.join(rootTarget, patient)):
			os.makedirs(os.path.join(rootTarget, patient))

		# Select thinnest series
		thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootSource, patient))]

		if len(thicknesses) == 0:
			print(patient + " no thick slice.")
			return

		min_thickness = min(thicknesses)

		# Collect all folder below patient folder which contains DICOM
		folder_dict = preprocessing_utils.CollectDICOMFolders(
			os.path.join(os.path.join(rootSource, patient), str(min_thickness)))

		if len(folder_dict.keys()) != 1:
			print(patient + ' more than 1 min thickness series.')
			return

		# Load image, store important tags
		for f in folder_dict:
			reader = sitk.ImageSeriesReader()
			series_found = reader.GetGDCMSeriesIDs(f)
			if len(series_found) != 1:
				print(patient + ' more series found.')
				return
			filenames = reader.GetGDCMSeriesFileNames(f, series_found[0])
			reader.SetFileNames(filenames)
			scan = reader.Execute()

		pixelSpacing = scan.GetSpacing()

		# if pixelSpacing[2] > 5.0:
		# 	print(patient)
		# 	return
		# else:
		# 	return

		# Save orientation
		direction = scan.GetDirection()
		origin = scan.GetOrigin()

		# Resample to unit pixelspacing
		scan, _ = preprocessing_utils.resample(sitk.GetArrayFromImage(scan),
											   np.array([pixelSpacing[2], pixelSpacing[0], pixelSpacing[1]]))
		scan = sitk.GetImageFromArray(scan)

		# Write resulting image and brain mask
		#		sitk.WriteImage(scan, os.path.join(rootTarget, patient, 'ScanResampled.mha'))

		# Set orientation back
		scan.SetDirection(direction)
		scan.SetOrigin(origin)

		# Segment brain
		scanBrain = os.path.join(rootTarget, patient, 'BrainMask.mha')
		if not os.path.exists(scanBrain):
			brain_mask = bs.segment_brain(scan, -20, 140, 160)
			sitk.WriteImage(brain_mask, scanBrain)
		else:
			brain_mask = sitk.ReadImage(scanBrain)

		if np.sum(sitk.GetArrayFromImage(brain_mask)) == 0:
			print(patient + " Skullstrip failed.")
			return

		# Align to center
		scan, brain_mask, _, _ = align.align(
			scan=scan,
			scanBrain=brain_mask,
			atlas=ATLAS,
			atlasBrain=BRAINATLAS,
			rootTarget=rootTarget,
			patient=patient,
			cta=False)

		sitk.WriteImage(scan, os.path.join(rootTarget, patient, 'AlignedScan.mha'))
		# sitk.WriteImage(brain_mask, os.path.join(rootTarget, 'AlignedBrainMask.mha'))

		# Save orientation
		direction = scan.GetDirection()
		origin = scan.GetOrigin()

		# Create skullstripped image
		input_data = sitk.GetArrayFromImage(scan)
		mask_data = sitk.GetArrayFromImage(brain_mask)
		brain_data = np.multiply(input_data, mask_data)
		skullstripped = sitk.GetImageFromArray(brain_data)

		skullstripped.SetSpacing(scan.GetSpacing())
		skullstripped.SetDirection(direction)
		skullstripped.SetOrigin(origin)

		sitk.WriteImage(skullstripped, os.path.join(rootTarget, patient, 'Skullstripped.mha'))

		print(patient + " done.")
	except:
		print(patient + " failed.")

def pipeline_CTA(patient, rootSource, rootTargetCTA, rootTargetMIP, rootTargetMIPBrain):
	try:
		if os.path.exists(
				os.path.join(rootTargetCTA, patient)):  # and len(os.listdir(os.path.join(rootTarget, patient))) == 0:
			return

		# Create output dict if not existing
		if not os.path.exists(os.path.join(rootTargetCTA, patient)):
			os.makedirs(os.path.join(rootTargetCTA, patient))
		if not os.path.exists(os.path.join(rootTargetMIP, patient)):
			os.makedirs(os.path.join(rootTargetMIP, patient))
		if not os.path.exists(os.path.join(rootTargetMIPBrain, patient)):
			os.makedirs(os.path.join(rootTargetMIPBrain, patient))

		# Select thinnest series
		thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootSource, patient))]

		if len(thicknesses) == 0:
			print(patient + " no thick slice.")
			return

		min_thickness = min(thicknesses)

		# Collect all folder below patient folder which contains DICOM
		folder_dict = preprocessing_utils.CollectDICOMFolders(
			os.path.join(os.path.join(rootSource, patient), str(min_thickness)))

		if len(folder_dict.keys()) != 1:
			print(patient + ' more than 1 min thickness series.')
			return

		# Load image, store important tags
		for f in folder_dict:
			reader = sitk.ImageSeriesReader()
			series_found = reader.GetGDCMSeriesIDs(f)
			if len(series_found) != 1:
				print(patient + ' more series found.')
				return
			filenames = reader.GetGDCMSeriesFileNames(f, series_found[0])
			reader.SetFileNames(filenames)
			scan = reader.Execute()

		pixelSpacing = scan.GetSpacing()

		# if pixelSpacing[2] > 5.0:
		# 	print(patient)
		# 	return
		# else:
		# 	return

		# Save orientation
		direction = scan.GetDirection()
		origin = scan.GetOrigin()

		# Resample to unit pixelspacing
		scan, _ = preprocessing_utils.resample(sitk.GetArrayFromImage(scan),
											   np.array([pixelSpacing[2], pixelSpacing[0], pixelSpacing[1]]))
		scan = sitk.GetImageFromArray(scan)

		# Set orientation back
		scan.SetDirection(direction)
		scan.SetOrigin(origin)

		# Write resulting image and brain mask
		sitk.WriteImage(scan, os.path.join(rootTargetCTA, patient, 'ScanResampled.mha'))

		# Segment brain
		scanBrain = os.path.join(rootTargetCTA, patient, 'BrainMask.mha')
		if not os.path.exists(scanBrain):
			brain_mask = bs.segment_brain(scan, -20, 330, 350)
		# sitk.WriteImage(brain_mask, scanBrain)
		else:
			brain_mask = sitk.ReadImage(scanBrain)

		if np.sum(sitk.GetArrayFromImage(brain_mask)) == 0:
			print(patient + " Skullstrip failed.")
			return

		# Align to center
		scan, brain_mask, _, _ = align.align(
			scan=scan,
			scanBrain=brain_mask,
			atlas=ATLAS,
			atlasBrain=BRAINATLAS,
			rootTarget=rootTargetCTA,
			patient=patient,
			cta=False)

		sitk.WriteImage(scan, os.path.join(rootTargetCTA, patient, 'AlignedScan.mha'))
		# sitk.WriteImage(brain_mask, os.path.join(rootTarget, 'AlignedBrainMask.mha'))

		# Compute MIP
		mip_img = mip.MIP(scan, 9, 3)
		sitk.WriteImage(mip_img, os.path.join(rootTargetMIP, patient, 'MIP.mha'))

		# Save orientation
		direction = scan.GetDirection()
		origin = scan.GetOrigin()
		spacing = scan.GetSpacing()

		# Create skullstripped image (again)
		input_data = sitk.GetArrayFromImage(scan)
		mask_data = sitk.GetArrayFromImage(brain_mask)
		brain_data = np.multiply(input_data, mask_data)
		scan = sitk.GetImageFromArray(brain_data)

		scan.SetSpacing(spacing)
		scan.SetDirection(direction)
		scan.SetOrigin(origin)

		sitk.WriteImage(scan, os.path.join(rootTargetCTA, patient, 'Skullstripped.mha'))

		# Compute MIP
		mip_img = mip.MIP(scan, 9, 3)
		sitk.WriteImage(mip_img, os.path.join(rootTargetMIPBrain, patient, 'MIP.mha'))

		print(patient + " done.")

	except:
		print(patient + " failed.")

def select_biggest_dimensions(patients, root):
	max_x = 0
	max_y = 0
	max_z = 0
	min_x = 1000
	min_y = 1000
	min_z = 1000

	for patient in patients:
		if not os.path.exists(os.path.join(root, patient, "MIP.mha")):
			print(patient)
			continue
		data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, patient, "MIP.mha")))
		coords = np.nonzero(data)
		if max(coords[2]) > max_x:
			max_x = max(coords[2])
			p_mx = patient
		if max(coords[1]) > max_y:
			max_y = max(coords[1])
			p_my = patient
		if max(coords[0]) > max_z:
			max_z = max(coords[0])
			p_mz = patient

		if min(coords[2]) < min_x:
			min_x = min(coords[2])
			p_ix = patient
		if min(coords[1]) < min_y:
			min_y = min(coords[1])
			p_iy = patient
		if min(coords[0]) < min_z:
			min_z = min(coords[0])
			p_iz = patient

	#	print(str(min_y) + " " + str(p_iy))
	#	print(str(max_z) + " " + str(p_mz))

	print(str(min_x) + " " + str(min_y) + " " + str(min_z))
	print(str(p_ix) + " " + str(p_iy) + " " + str(p_iz))
	print(str(max_x) + " " + str(max_y) + " " + str(max_z))
	print(str(p_mx) + " " + str(p_my) + " " + str(p_mz))

def resize_image(patient, SIZE, rootSource, rootTarget):
	try:
		if os.path.exists(os.path.join(rootTarget, patient + ".mha")):
			return
		if len(os.listdir(os.path.join(rootSource, patient))) == 0:
			shutil.rmtree(os.path.join(rootSource, patient))

		img = sitk.ReadImage(os.path.join(rootSource, patient, "MIP.mha"))
		data = sitk.GetArrayFromImage(img)
		new_data = np.zeros(SIZE)
		if data.shape[0] > SIZE[0] and data.shape[1] < SIZE[1]:
			new_data[
			:,
			int(np.floor((SIZE[1] - data.shape[1]) / 2)): SIZE[1] - int(
				np.ceil((SIZE[1] - data.shape[1]) / 2)),
			int(np.floor((SIZE[2] - data.shape[2]) / 2)): SIZE[2] - int(
				np.ceil((SIZE[2] - data.shape[2]) / 2))
			] = data[int(np.floor((data.shape[0] - SIZE[0]) / 2)):
					data.shape[0] - int(np.ceil((data.shape[0] - SIZE[0]) / 2)), :, :]
		if data.shape[1] > SIZE[1] and data.shape[0] < SIZE[0]:
			new_data[
			int(np.floor((SIZE[0] - data.shape[0]) / 2)): SIZE[0] - int(
				np.ceil((SIZE[0] - data.shape[0]) / 2)),
			:, :
			] = data[:,
					int(np.floor((data.shape[1] - SIZE[1]) / 2)):
						data.shape[1] - int(np.ceil((data.shape[1] - SIZE[1]) / 2)),
					int(np.floor((data.shape[2] - SIZE[2]) / 2)):
						data.shape[2] - int(np.ceil((data.shape[2] - SIZE[2]) / 2))
					]
		if data.shape[1] > SIZE[1] and data.shape[0] > SIZE[0]:
			new_data = data[
				int(np.floor((data.shape[0] - SIZE[0]) / 2)):
					data.shape[0] - int(np.ceil((data.shape[0] - SIZE[0]) / 2)),
				int(np.floor((data.shape[1] - SIZE[1]) / 2)):
					data.shape[1] - int(np.ceil((data.shape[1] - SIZE[1]) / 2)),
				int(np.floor((data.shape[2] - SIZE[2]) / 2)):
					data.shape[2] - int(np.ceil((data.shape[2] - SIZE[2]) / 2))
				]
		if data.shape[1] < SIZE[1] and data.shape[0] < SIZE[0]:
			new_data[
			int(np.floor((SIZE[0] - data.shape[0]) / 2)): SIZE[0] - int(
				np.ceil((SIZE[0] - data.shape[0]) / 2)),
			int(np.floor((SIZE[1] - data.shape[1]) / 2)): SIZE[1] - int(
				np.ceil((SIZE[1] - data.shape[1]) / 2)),
			int(np.floor((SIZE[2] - data.shape[2]) / 2)): SIZE[2] - int(
				np.ceil((SIZE[2] - data.shape[2]) / 2))
			] = data
		new_img = sitk.Cast(sitk.GetImageFromArray(new_data), sitk.sitkInt16)
		new_img.SetSpacing(img.GetSpacing())
		#		new_img.AddMetaData("0010|0020", patient)

		#		writer = sitk.ImageFileWriter()
		#		writer.SetFileName(os.path.join(rootTarget, patient + ".dcm"))
		#		writer.Execute(new_img)
		sitk.WriteImage(new_img, os.path.join(rootTarget, patient + ".mha"))

	except:
		print(patient + " failed.")

def copy_resampled_by_slice_thickness(patient, threshold, rootOriginal, rootSource, rootTarget):
	# Select thinnest series
	thicknesses = [float(thickness) for thickness in os.listdir(os.path.join(rootOriginal, patient))]

	if len(thicknesses) == 0:
		print(patient + " no thick slice.")
		return

	min_thickness = min(thicknesses)
	print(str(min_thickness))

	if min_thickness > threshold:
		return

	preprocessing_utils.CopyDirectory(os.path.join(rootSource, patient), rootTarget)


def remove_MIP(patient, rootSource):
	#	try:
	# Collect all folder below patient folder which contains DICOM
	folder_dict = preprocessing_utils.CollectDICOMFolders(os.path.join(rootSource, patient))
	for folder in folder_dict:
		dicomFiles = folder_dict[folder]

		if "MIP" in preprocessing_utils.GetSeriesDescription(dicomFiles[0]) or \
				"mip" in preprocessing_utils.GetSeriesDescription(dicomFiles[0]) or \
				"MIP" in preprocessing_utils.GetImageType(dicomFiles[0]) or \
				"mip" in preprocessing_utils.GetImageType(dicomFiles[0]):
			shutil.rmtree(folder)
			print(patient + " deleted.")


#	except:
#		print(patient + " failed.")

def remove_badCT(patient, rootSource):
	#	try:
	# Collect all folder below patient folder which contains DICOM
	folder_dict = preprocessing_utils.CollectDICOMFolders(os.path.join(rootSource, patient))
	for folder in folder_dict:
		dicomFiles = folder_dict[folder]

		if "H20f" in preprocessing_utils.GetSeriesDescription(dicomFiles[0]) or \
				"h20f" in preprocessing_utils.GetSeriesDescription(dicomFiles[0]) or \
				"H20f" in preprocessing_utils.GetConvKernel(dicomFiles[0]) or \
				"h20f" in preprocessing_utils.GetConvKernel(dicomFiles[0]):
			shutil.rmtree(folder)


def skullstrip(patient, rootSource, rootTarget):
#	try:
		scan = sitk.ReadImage(os.path.join(rootSource, patient))
		scanBrain = os.path.join(rootTarget, patient)
		if not os.path.exists(scanBrain):
			brain_mask = bs.segment_brain(scan, -20, 330, 350)

		if np.sum(sitk.GetArrayFromImage(brain_mask)) == 0:
			print(patient + " Skullstrip failed.")
			return

		# Save orientation
		direction = scan.GetDirection()
		origin = scan.GetOrigin()
		spacing = scan.GetSpacing()

		# Create skullstripped image (again)
		input_data = sitk.GetArrayFromImage(scan)
		mask_data = sitk.GetArrayFromImage(brain_mask)
		brain_data = np.multiply(input_data, mask_data)
		scan = sitk.GetImageFromArray(brain_data)

		scan.SetSpacing(spacing)
		scan.SetDirection(direction)
		scan.SetOrigin(origin)

		sitk.WriteImage(scan, scanBrain)

		print(patient + " done.")
#	except:
#		print(patient + " failed.")


ATLAS = r"E:/Atlases/MNI_atlas.nii"
BRAINATLAS = r"E:/Atlases/MNI_brainmask.nii"
FAILED_LIST = []

if __name__ == '__main__':
	#	patients = os.listdir(NCCT_THIN)
	#	for patient in patients:
	#		pipeline_NCCT(patient, NCCT_ST, NCCT_THINNEST)
	#	select_biggest_dimensions(patients, NCCT_THIN)

	# Retry failed patients
	# 	patients = os.listdir(NCCT_ST)
	# 	for patient in patients:
	# 		if os.path.exists(os.path.join(NCCT_THINNEST, patient)) and len(os.listdir(os.path.join(NCCT_THINNEST, patient))) == 0:
	# #			FAILED_LIST.append(patient)
	# 			shutil.rmtree(os.path.join(NCCT_THINNEST, patient))
	#
	# 	for patient in FAILED_LIST:
	# 		print(patient)
	# 		pipeline_NCCT(patient, NCCT_ST, NCCT_THINNEST)
	# 		print(patient + " done.")

	# Try only 1 patient
	# start = time.time()
	# pipeline_NCCT('R0027', TEST_REG, TEST_REG_OUT)
	# end = time.time()
	# print(end - start)

	# Run pipeline for whole dataset
	#		patients = os.listdir(CTA_ST)
	#		with Pool() as p:
	#			p.starmap(pipeline_NCCT, zip(patients, repeat(CTA_ST), repeat(CTA_ST)))

	# Run resize for whole dataset
	#	patients = os.listdir(NCCT_THINNEST)
	#	NCCT_THINNEST_SIZE = (252, 336, 336)
	#	NCCT_THIN_SIZE = (257, 336, 336)
	#	resize_image('R0001', NCCT_THINNEST_SIZE, NCCT_THINNEST, NCCT_THINNEST_RESIZED)
	#	with Pool() as p:
	#		p.starmap(resize_image, zip(patients, repeat(NCCT_THINNEST_SIZE), repeat(NCCT_THINNEST), repeat(NCCT_THINNEST_RESIZED)))

	# Select thinner slices than 3.0
	# 	patients = os.listdir(NCCT_THINNEST)
	# 	threshold = 3.0
	# #	copy_resampled_by_slice_thickness('R0001', threshold, NCCT_ST, NCCT_THINNEST, NCCT_THIN)
	# 	with Pool() as p:
	# 		p.starmap(copy_resampled_by_slice_thickness, zip(patients, repeat(threshold), repeat(NCCT_ST), repeat(NCCT_THINNEST), repeat(NCCT_THIN)))

	# Select soft tissue for CTAs
	#	patients = os.listdir(CTA_ST)
	#	softtissue_selection("R0001", CTA_REGISTRY, CTA_ST)
	#	with Pool() as p:
	#		p.starmap(softtissue_selection, zip(patients, repeat(CTA_REGISTRY), repeat(CTA_ST)))

	# Remove MIPs from CTA series
	patients = os.listdir(CTA_THINNEST_RESIZED)
#	print(patients[0])
#	start = time.time()
#	mip_image = mip.MIP_2D(sitk.ReadImage(os.path.join(CTA_THINNEST_RESIZED, patients[0])), 20)
#	sitk.WriteImage(mip_image, os.path.join(CTA_2D_MIP_RESIZED, patients[0]))
#	end = time.time()
#	print(end - start)

	with Pool() as p:
		p.starmap(mip.MIP_2D, zip(repeat(CTA_THINNEST_RESIZED), patients, repeat(CTA_2D_MIP_RESIZED)))

	# Run pipeline for whole dataset
	#		start = time.time()
	#	patients = os.listdir(CTA_ST)
	#		pipeline_CTA("R1744", CTA_ST, CTA_THINNEST, CTA_MIP, CTA_SKULLSTRIPPED)
	#		end = time.time()
	#		print(end - start)

	#	with Pool() as p:
	#		p.starmap(pipeline_CTA, zip(patients, repeat(CTA_ST), repeat(CTA_THINNEST), repeat(CTA_MIP), repeat(CTA_SKULLSTRIPPED)))

	# Select biggest dimensions
	#	patients = os.listdir(CTA_SKULLSTRIPPED)
	#	select_biggest_dimensions(patients, CTA_SKULLSTRIPPED)

	# Run resize for whole dataset
#	patients = os.listdir(CTA_THINNEST)
#	CTA_THINNEST_SIZE = (318, 433, 433)
#	CTA_MIP_SIZE = (67, 495, 495)
#	CTA_MIP_SKULL_SIZE = (53, 432, 432)
#	resize_image('R1536', CTA_THINNEST_SIZE, CTA_THINNEST, CTA_THINNEST_RESIZED)
#	with Pool() as p:
#		p.starmap(resize_image, zip(patients, repeat(CTA_MIP_SKULL_SIZE), repeat(CTA_SKULLSTRIPPED), repeat(CTA_SKULLSTRIPPED_RESIZED)))

	# Skullstripp MIPs
#	patients = os.listdir(CTA_MIP_RESIZED)
#	skullstrip("R0001", CTA_MIP_RESIZED, CTA_SKULLSTRIPPED_RESIZED)
#	with Pool() as p:
#		p.starmap(skullstrip, zip(patients, repeat(CTA_MIP_RESIZED), repeat(CTA_SKULLSTRIPPED_RESIZED)))

#	root = '/home/hilbysfe/DATA/SUPERVISED/REGISTRY/NCCT/ASPECTS_ALL_PART1/'
#	rootTarget = '/home/hilbysfe/DATA/SUPERVISED/REGISTRY/NCCT/ASPECTS_FAILED'
#	patients = os.listdir(root)
#	for patient in patients:
#		thicknesses = os.listdir(os.path.join(root,patient))
#		if len(thicknesses) == 0:
#		#			shutil.rmtree(os.path.join(root,patient))
#			print(patient + ' to be deleted.')
#		for thickness in thicknesses:
#			if not os.path.exists(os.path.join(root,patient,thickness,'BsplineRegisteredASPECTS.mha')):
#				if not os.path.exists(os.path.join(rootTarget,patient,thickness)):
#					os.makedirs(os.path.join(rootTarget,patient,thickness))
#					shutil.copytree(os.path.join(root,patient,thickness), os.path.join(rootTarget,patient,thickness))
#					shutil.rmtree(os.path.join(root,patient,thickness))
#					print(patient + ' copied.')
#				else:
#				#					shutil.rmtree(os.path.join(root,patient,thickness))
#					print(patient + ' in failed.')

#	patients = os.listdir(CTA_THINNEST)
#	for patient in patients:
#		thicknesses = [float(thick) for thick in os.listdir(os.path.join(CTA_ST, patient))]
#		print(min(thicknesses))