import SimpleITK as sitk
import math
import numpy as np
from preprocessing_utils import CollectDICOMFolders
import os
from multiprocessing import Pool


rootSource = 'E:\\REGISTRY\\CTA_THIN'
rootTarget = 'E:\\REGISTRY\\CTA_THIN_SKULLSTRIPPED'


# ============ FOR CTA ============
# Radius (in mm) of biggest hole in the skull after the foramen magnum.
FORAMEN_RADIUS_IN_MM = 7

MIN_SKULL_HU_VALUE = 350
MAX_BRAIN_HU_VALUE = 330
MIN_BRAIN_HU_VALUE = -20

# ============ FOR NCCT ============
# FORAMEN_RADIUS_IN_MM = 7
#
# MIN_SKULL_HU_VALUE = 160
# MAX_BRAIN_HU_VALUE = 140
# MIN_BRAIN_HU_VALUE = -20

def strip(input_image):
	FORAMEN_RADIUS_3D = (math.floor(FORAMEN_RADIUS_IN_MM / input_image.GetSpacing()[0]),
						 math.floor(FORAMEN_RADIUS_IN_MM / input_image.GetSpacing()[1]),
						 math.floor(FORAMEN_RADIUS_IN_MM / input_image.GetSpacing()[2]))

	# Segment all bone in the image.
	aux_image = input_image > MIN_SKULL_HU_VALUE

	# Bring values for the scale from 0 to 1 (0 is background and 1 is the foreground).
	clamp = sitk.ClampImageFilter()
	clamp.SetLowerBound(0)
	clamp.SetUpperBound(1)
	aux_image = clamp.Execute(aux_image)
	aux_image = sitk.Cast(aux_image, sitk.sitkInt32)

	# Close skull holes
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetBackgroundValue(0)
	dilate.SetForegroundValue(1)
	dilate.SetKernelRadius(FORAMEN_RADIUS_3D)
	aux_image = dilate.Execute(aux_image)
	skull_without_holes = aux_image

	# Segment area outside the skull. Also finds the slice with seed for brain segmentation inside the skull.
	slice_id = input_image.GetSize()[2] - 1
	while slice_id >= 0:
		aux_slice = aux_image[:, :, slice_id]

		region_growing_filter = sitk.ConnectedThresholdImageFilter()
		region_growing_filter.SetLower(0)
		region_growing_filter.SetUpper(0)
		region_growing_filter.SetSeed([0, 0])

		slice_mask = region_growing_filter.Execute(aux_slice)
		slice_mask = sitk.Cast(slice_mask, aux_image.GetPixelID())
		slice_mask = sitk.JoinSeries(slice_mask)

		aux_image = sitk.Paste(aux_image, slice_mask, slice_mask.GetSize(), destinationIndex=[0, 0, slice_id])
		slice_id = slice_id - 1

	# Dilate the segmentation of the area outside the skull back to the skull border.
	outside_skull_mask = dilate.Execute(aux_image)
	outside_skull_mask.SetOrigin(input_image.GetOrigin())
	outside_skull_mask.SetDirection(input_image.GetDirection())
	outside_skull_mask.SetSpacing(input_image.GetSpacing())

	# Save intermediate result. TODO: To be removed.
	# sitk.WriteImage(outside_skull_mask, "C:/Users/renan/Downloads/outside_skull_mask.mha")
	# sitk.WriteImage(skull_without_holes, "C:/Users/renan/Downloads/skull_without_holes.mha")

	# Remove other connected components that are not part of the brain.
	outside_brain_mask = outside_skull_mask + skull_without_holes
	outside_brain_mask = sitk.Clamp(outside_brain_mask, lowerBound=0, upperBound=1)
	outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)
	outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
	outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)
	outside_brain_mask = sitk.RelabelComponent(sitk.ConnectedComponent(outside_brain_mask))
	outside_brain_mask = sitk.Threshold(outside_brain_mask)

	# Dilate the segmentation of the area inside the skull back to the skull border.
	outside_brain_mask = dilate.Execute(outside_brain_mask)
	outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
	outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)

	# Save intermediate result. TODO: To be removed.
	# sitk.WriteImage(outside_brain_mask, "C:/Users/renan/Downloads/outside_brain_mask.mha")

	# Finds slice with biggest portion of brain.

	previous_area = -1
	previous_slice_id = -1
	slices_where_area_trend_change = []
	slice_id = input_image.GetSize()[2] - 1
	while slice_id >= 0:
		slice_mask = outside_brain_mask[:, :, slice_id]
		slice_mask = sitk.Cast(slice_mask, sitk.sitkInt32)
		label_info_filter = sitk.LabelStatisticsImageFilter()
		label_info_filter.Execute(slice_mask, slice_mask)
		if previous_area == -1:
			previous_area = label_info_filter.GetCount(0)
			previous_slice_id = slice_id
		else:
			current_area = label_info_filter.GetCount(0)
			if abs(current_area - previous_area) > 0.05 * (
					current_area if current_area < previous_area else previous_area):
				if current_area < previous_area:
					slices_where_area_trend_change.append(previous_slice_id)
			previous_area = current_area
			previous_slice_id = slice_id
		slice_id = slice_id - 1

	# HU value of air + threshold to account for noise + value to stay above the max brain HU value.
	HU_DELTA = 1000 + 100 + (MAX_BRAIN_HU_VALUE + abs(MIN_BRAIN_HU_VALUE))
	aux_image = outside_brain_mask * HU_DELTA + sitk.Cast(input_image, sitk.sitkInt32)
	aux_image = sitk.Cast(aux_image, sitk.sitkInt32)

	# Save intermediate result. TODO: To be removed.
	# sitk.WriteImage(aux_image, "C:/Users/renan/Downloads/region_growing_input.mha")

	# Get a seed inside the brain
	slice_with_big_brain_area_number = 0
	if len(slices_where_area_trend_change) != 0:
		slice_with_big_brain_area_number = slices_where_area_trend_change[0]

	slice_with_big_brain_area = outside_brain_mask[:, :, slice_with_big_brain_area_number]
	slice_with_big_brain_area = sitk.InvertIntensity(slice_with_big_brain_area, 1)
	slice_info = sitk.LabelStatisticsImageFilter()
	slice_info.Execute(slice_with_big_brain_area, slice_with_big_brain_area)
	bound_box = slice_info.GetBoundingBox(1)
	seed_x = int((bound_box[1] - bound_box[0]) / 2 + bound_box[0])
	seed_y = int((bound_box[3] - bound_box[2]) / 2 + bound_box[2])
	seed = (seed_x, seed_y, slice_with_big_brain_area_number)

	# Use region growing inside the brain
	region_growing_filter = sitk.ConnectedThresholdImageFilter()
	region_growing_filter.SetLower(MIN_BRAIN_HU_VALUE)
	region_growing_filter.SetUpper(MAX_BRAIN_HU_VALUE)
	region_growing_filter.SetSeed(seed)
	aux_image = region_growing_filter.Execute(aux_image)
	aux_image = sitk.BinaryFillhole(aux_image)
	aux_image = sitk.Cast(aux_image, sitk.sitkUInt8)

	# Save intermediate result. TODO: To be removed.
	# sitk.WriteImage(aux_image, "C:/Users/renan/Downloads/skull stripping validation data/region_growing_output.mha")

	FINAL_KERNEL = (math.floor(1 / input_image.GetSpacing()[0]),
					math.floor(1 / input_image.GetSpacing()[1]),
					math.floor(1 / input_image.GetSpacing()[2]))
	erode = sitk.BinaryErodeImageFilter()
	erode.SetForegroundValue(1)
	erode.SetBackgroundValue(0)
	erode.SetKernelRadius(FINAL_KERNEL)

	aux_image = erode.Execute(aux_image)

	FINAL_KERNEL = (math.floor(2 / input_image.GetSpacing()[0]),
					math.floor(2 / input_image.GetSpacing()[1]),
					math.floor(2 / input_image.GetSpacing()[2]))
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetForegroundValue(1)
	dilate.SetBackgroundValue(0)
	dilate.SetKernelRadius(FINAL_KERNEL)

	aux_image = dilate.Execute(aux_image)
	aux_image = erode.Execute(aux_image)

	# Segment all bone in the image but now using the brain threshold.
	bone_mask = input_image > MAX_BRAIN_HU_VALUE

	# Bring values for the scale from 0 to 1 (0 is background and 1 is the foreground).
	clamp = sitk.ClampImageFilter()
	clamp.SetLowerBound(0)
	clamp.SetUpperBound(1)
	bone_mask = clamp.Execute(bone_mask)
	bone_mask = sitk.Cast(bone_mask, sitk.sitkInt32)

	aux_image = sitk.Cast(aux_image, sitk.sitkInt32)

	aux_image = aux_image - bone_mask
	aux_image = clamp.Execute(aux_image)
	aux_image = sitk.Cast(aux_image, sitk.sitkInt32)

	aux_image = sitk.RelabelComponent(sitk.ConnectedComponent(aux_image))
	aux_image = sitk.Threshold(aux_image)
	aux_image = sitk.BinaryFillhole(aux_image)

	aux_image = sitk.Cast(aux_image, sitk.sitkUInt8)

	return aux_image

def skullstripping(patient):
	if not os.path.exists(os.path.join(rootTarget, patient)):
		try:
			dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
			if len(dicomFolders) != 1:
				print(patient + ' more dicom folder found')
				return
			for f in dicomFolders:
				# Read input image
				reader = sitk.ImageSeriesReader()
				series_found = reader.GetGDCMSeriesIDs(f)

				if len(series_found) != 1:
					print(patient + ' more series found.')
					return

				reader.SetFileNames(reader.GetGDCMSeriesFileNames(f, series_found[0]))
				input_image = reader.Execute()

				aux_image = strip(input_image)

				input_data = sitk.GetArrayFromImage(input_image)
				aux_data = sitk.GetArrayFromImage(aux_image)
				brain_data = np.multiply(input_data, aux_data)
				brain_image = sitk.GetImageFromArray(brain_data)

				brain_image.SetSpacing(input_image.GetSpacing())

				# Save final result
				if not os.path.exists(os.path.join(rootTarget, patient)):
					os.makedirs(os.path.join(rootTarget, patient))

				sitk.WriteImage(brain_image, os.path.join(rootTarget, patient, patient + '.mha'))

				print(patient + " done.")
		except:
			print(patient + ' failed.')



if __name__ == '__main__':
	patients = os.listdir(rootSource)
	for patient in patients:
		if os.path.exists(os.path.join(rootTarget, patient)):
			continue
		skullstripping(patient)
