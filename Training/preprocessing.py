import numpy as np
import SimpleITK as sitk
import os
import time
import shutil
from multiprocessing import Pool
from itertools import repeat

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

CTA_REGISTRY = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_THIN'
CTA_REGISTRY_RESAMPLED = r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_THIN_RESAMPLED'

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

	return 'h31s' in kernel or \
		   'H31s' in kernel or \
		   'J40s' in kernel or \
		   'h31s' in series or \
		   'H31s' in series or \
		   'J40s' in series or \
		   (preprocessing_utils.GetWindowCenter(file) < 100.0 and preprocessing_utils.GetWindowWidth(file) < 200.0)


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


def softtissue_selection(patient, threshold):
	try:
		#		if os.path.exists(os.path.join(NCCT_THIN, patient)):
		#			print(patient + ' done.')
		#			return

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
			#			print(os.path.join(NCCT_ST, patient, str(sliceMap[folder])))
			if not os.path.exists(os.path.join(NCCT_ST, patient, str(sliceMap[folder]))):
				preprocessing_utils.CopyFiles(baselines[folder],
											  os.path.join(NCCT_ST_NEW, patient, str(sliceMap[folder])))

		print(patient + ' done.')
	except Exception as e:
		print(patient + ' failed: %s' % e)


def pipeline_NCCT(patient, rootSource, rootTarget):
#	try:
#		if os.path.exists(os.path.join(rootTarget, patient)): # and len(os.listdir(os.path.join(rootTarget, patient))) == 0:
#			return

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
	# except:
	# 	print(patient + " failed.")


def pipeline_CTA(patient, rootSource, rootTarget):
#	try:
#		if os.path.exists(os.path.join(rootTarget, patient)): # and len(os.listdir(os.path.join(rootTarget, patient))) == 0:
#			return

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
			brain_mask = bs.segment_brain(scan, -20, 330, 350)
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
	# except:
	# 	print(patient + " failed.")


def select_biggest_dimensions(patients, root):
	max_x = 0
	max_y = 0
	max_z = 0
	min_x = 1000
	min_y = 1000
	min_z = 1000

	for patient in patients:
		if not os.path.exists(os.path.join(root, patient, "Skullstripped.mha")):
			print(patient)
			continue
		data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, patient, "Skullstripped.mha")))
		coords = np.nonzero(data)
		# if max(coords[2]) > max_x:
		# 	max_x = max(coords[2])
		# 	p_mx = patient
		# if max(coords[1]) > max_y:
		# 	max_y = max(coords[1])
		# 	p_my = patient
		# if max(coords[0]) > max_z:
		# 	max_z = max(coords[0])
		# 	p_mz = patient

		if min(coords[2]) < min_x:
			min_x = min(coords[2])
			p_ix = patient
		# if min(coords[1]) < min_y:
		# 	min_y = min(coords[1])
		# 	p_iy = patient
		# if min(coords[0]) < min_z:
		# 	min_z = min(coords[0])
		# 	p_iz = patient

	print(str(min_x) + " " + str(p_ix))

	# print(str(min_x) + " " + str(min_y) + " " + str(min_z))
	# print(str(p_ix) + " " + str(p_iy) + " " + str(p_iz))
	# print(str(max_x) + " " + str(max_y) + " " + str(max_z))
	# print(str(p_mx) + " " + str(p_my) + " " + str(p_mz))


def resize_image(patient, NCCT_THINNEST_SIZE, rootSource, rootTarget):
#	try:
		if os.path.exists(os.path.join(rootTarget, patient + ".mha")):
			return
		img = sitk.ReadImage(os.path.join(rootSource, patient, "Skullstripped.mha"))
		data = sitk.GetArrayFromImage(img)[:,::-1,:]
		new_data = np.zeros(NCCT_THINNEST_SIZE)
		if data.shape[0] > NCCT_THINNEST_SIZE[0] and data.shape[1] < NCCT_THINNEST_SIZE[1]:
			new_data[
				:,
				int(np.floor((NCCT_THINNEST_SIZE[1] - data.shape[1]) / 2)): NCCT_THINNEST_SIZE[1] - int(
					np.ceil((NCCT_THINNEST_SIZE[1] - data.shape[1]) / 2)),
				int(np.floor((NCCT_THINNEST_SIZE[2] - data.shape[2]) / 2)): NCCT_THINNEST_SIZE[2] - int(
					np.ceil((NCCT_THINNEST_SIZE[2] - data.shape[2]) / 2))
			] = data[int(data.shape[0]/2)-int(np.floor(NCCT_THINNEST_SIZE[0]/2)):int(data.shape[0]/2)+int(np.ceil(NCCT_THINNEST_SIZE[0]/2)),:,:]
		if data.shape[1] > NCCT_THINNEST_SIZE[1] and data.shape[0] < NCCT_THINNEST_SIZE[0]:
			new_data[
				int(np.floor((NCCT_THINNEST_SIZE[0] - data.shape[0]) / 2)): NCCT_THINNEST_SIZE[0] - int(np.ceil((NCCT_THINNEST_SIZE[0] - data.shape[0]) / 2)),
				:, :
			] = data[:,
					int(data.shape[1] / 2) - int(np.floor(NCCT_THINNEST_SIZE[1] / 2)):int(
						np.ceil(data.shape[1] / 2) + int(NCCT_THINNEST_SIZE[1] / 2)),
					int(data.shape[2] / 2) - int(np.floor(NCCT_THINNEST_SIZE[2] / 2)):int(
						np.ceil(data.shape[2] / 2) + int(NCCT_THINNEST_SIZE[2] / 2))
				]
		if data.shape[1] > NCCT_THINNEST_SIZE[1] and data.shape[0] > NCCT_THINNEST_SIZE[0]:
			new_data = data[
							int(data.shape[0] / 2) - int(np.floor(NCCT_THINNEST_SIZE[0] / 2)):int(
								np.ceil(data.shape[0] / 2) + int(NCCT_THINNEST_SIZE[0] / 2)),
							int(data.shape[1] / 2) - int(np.floor(NCCT_THINNEST_SIZE[1] / 2)):int(
								np.ceil(data.shape[1] / 2) + int(NCCT_THINNEST_SIZE[1] / 2)),
							int(data.shape[2] / 2) - int(np.floor(NCCT_THINNEST_SIZE[2] / 2)):int(
								np.ceil(data.shape[2] / 2) + int(NCCT_THINNEST_SIZE[2] / 2))
					]
		if data.shape[1] < NCCT_THINNEST_SIZE[1] and data.shape[0] < NCCT_THINNEST_SIZE[0]:
			new_data[
				int(np.floor((NCCT_THINNEST_SIZE[0] - data.shape[0]) / 2)): NCCT_THINNEST_SIZE[0] - int(np.ceil((NCCT_THINNEST_SIZE[0] - data.shape[0]) / 2)),
				int(np.floor((NCCT_THINNEST_SIZE[1] - data.shape[1]) / 2)): NCCT_THINNEST_SIZE[1] - int(np.ceil((NCCT_THINNEST_SIZE[1] - data.shape[1]) / 2)),
				int(np.floor((NCCT_THINNEST_SIZE[2] - data.shape[2]) / 2)): NCCT_THINNEST_SIZE[2] - int(np.ceil((NCCT_THINNEST_SIZE[2] - data.shape[2]) / 2))
			] = data
		new_img = sitk.Cast(sitk.GetImageFromArray(new_data), sitk.sitkInt16)
		new_img.SetSpacing(img.GetSpacing())
#		new_img.AddMetaData("0010|0020", patient)

#		writer = sitk.ImageFileWriter()
#		writer.SetFileName(os.path.join(rootTarget, patient + ".dcm"))
#		writer.Execute(new_img)
		sitk.WriteImage(new_img, os.path.join(rootTarget, patient + ".mha"))
#	except:
#		print(patient + " failed.")

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
#	patients = os.listdir(TEST_REG)
#	with Pool() as p:
#		p.starmap(pipeline_NCCT, zip(patients, repeat(TEST_REG), repeat(TEST_REG_OUT)))

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
