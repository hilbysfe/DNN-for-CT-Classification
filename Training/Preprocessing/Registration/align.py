import SimpleITK as sitk

import math
import numpy
import os
from datetime import datetime
import argparse

def _find_main_axis(image):
    non_zero_elements = numpy.nonzero(sitk.GetArrayFromImage(image))
    x = non_zero_elements[0]
    y = non_zero_elements[1]

    x = x - numpy.mean(x)
    y = y - numpy.mean(y)
    coords = numpy.vstack([x, y])

    cov = numpy.cov(coords)
    evals, evecs = numpy.linalg.eig(cov)

    sort_indices = numpy.argsort(evals)[::-1]
    axis1, axis2 = evecs[:, sort_indices]
    x1, y1 = axis1  # Eigenvector with largest eigenvalue
    x2, y2 = axis2
    return x1, y1, x2, y2

def align(scan, scanBrain, atlas, atlasBrain, cta = False):

	#if scanBrain == "":
	#	scanBrain = os.path.join(scanFolder, "BrainMask.mha")
	#	if not os.path.exists(scanBrain):
	#		# Skull stripping part.
	#		image = sitk.ReadImage(scan)
	#		if cta:
	#			image = bs.segment_brain(image, -20, 330, 350)
	#		else:
	#			image = bs.segment_brain(image, -20, 140, 160)		
	#		sitk.WriteImage(image, scanBrain)

	# Load atlas and atlas brain mask.
	atlas = sitk.Cast(sitk.ReadImage(atlas), sitk.sitkFloat32)
	atlas_brain_mask = sitk.ReadImage(atlasBrain)
	
	# Resample atlas to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(atlas)
	out_size = scan.GetSize()
	in_size = atlas.GetSize()
	resample.SetSize(out_size)
	in_spacing = atlas.GetSpacing()
	outputspacing = [in_spacing[0] * (in_size[0]/out_size[0]), in_spacing[1] * (in_size[1]/out_size[1]), in_spacing[2] * (in_size[2]/out_size[2])]
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas = resample.Execute(atlas)	

	#sitk.WriteImage(atlas, os.path.join(scanFolder, "ResampledAtlas.mha"))

	# Resample atlas_mask to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(atlas_brain_mask)
	resample.SetSize(out_size)
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas_brain_mask = resample.Execute(atlas_brain_mask)	

	#sitk.WriteImage(atlas_brain_mask, os.path.join(scanFolder, "ResampledAtlasBrainMask.mha"))
		
	# Initial transform to align images based on center of mass.
	initial_transform = sitk.CenteredTransformInitializer(atlas_brain_mask,  # fixed image
															scanBrain,  # moving image
															sitk.Euler3DTransform(),
															sitk.CenteredTransformInitializerFilter.MOMENTS)

	moving_image_mask = sitk.Resample(scanBrain,  # moving image
										atlas_brain_mask,  # fixed image
										initial_transform,
										sitk.sitkNearestNeighbor,
										0.0,
										sitk.sitkUInt8)
	
	composed_transformation = sitk.Transform(3, sitk.sitkComposite)
	composed_transformation.AddTransform(initial_transform)

	shape_info = sitk.LabelShapeStatisticsImageFilter()
	shape_info.Execute(moving_image_mask)
	centroid = shape_info.GetCentroid(1)
	centroid_index = sitk.Image.TransformPhysicalPointToContinuousIndex(moving_image_mask, centroid)
	# sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "BeforeInitialAlignmentBrainMask.mha"))
	
	# Align Z plane using PCA.
	#plane_image = moving_image_mask[:, :, int(centroid_index[2])]
	#axes = _find_main_axis(plane_image)
	#rotation_info = (axes[0], axes[1]) if abs(axes[0]) > abs(axes[2]) else (axes[2], axes[3])
	#rotation_direction = rotation_info[0] * rotation_info[1]
	#rotation_direction = 1.0 if rotation_direction > 0 else -1.0
	#rotation_angle = rotation_direction * math.acos(abs(rotation_info[0]))
	#rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, rotation_angle)
	#moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	#composed_transformation.AddTransform(rotation)
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskZ.mha"))

	reference_plane = int(centroid_index[2])
	plane_image = moving_image_mask[:, :, reference_plane]
	angle = -1.3  # Approximately -75 degrees.
	best_angle = 0.0
	step = 0.01
	image_width = plane_image.GetSize()[0]
	half_image_width = int(image_width / 2)
	min_diff = image_width * plane_image.GetSize()[1]
	rotation_center = (centroid[0], centroid[1])
	while angle <= 1.3:
		rotation2d = sitk.Euler2DTransform((centroid[0], centroid[1]), angle)
		rotated_plane = sitk.Resample(plane_image, plane_image, rotation2d, sitk.sitkNearestNeighbor, 0.0)
		left_side = sitk.GetArrayFromImage(rotated_plane[0:half_image_width, :])
		right_side = sitk.GetArrayFromImage(rotated_plane[half_image_width:2*half_image_width, :])
		right_side = numpy.flip(right_side, 1)
		diff_sides = numpy.square(right_side - left_side)
		count_diff = numpy.count_nonzero(diff_sides)
		if count_diff < min_diff:
			best_angle = angle
			min_diff = count_diff
		angle += step
	rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, best_angle)
	moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	composed_transformation.AddTransform(rotation)
	# sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskZ.mha"))

	# Align Y plane using symmetry detection.
	reference_plane = int(centroid_index[1])
	plane_image = moving_image_mask[:, reference_plane, :]
	angle = -0.8  # Approximately -45 degrees.
	best_angle = 0.0
	step = 0.01
	image_width = plane_image.GetSize()[0]
	half_image_width = int(image_width / 2)
	min_diff = image_width * plane_image.GetSize()[1]
	rotation_center = (centroid[0], centroid[2])
	while angle <= 0.8:
		rotation2d = sitk.Euler2DTransform((centroid[0], centroid[2]), angle)
		rotated_plane = sitk.Resample(plane_image, plane_image, rotation2d, sitk.sitkNearestNeighbor, 0.0)
		left_side = sitk.GetArrayFromImage(rotated_plane[0:half_image_width, :])
		right_side = sitk.GetArrayFromImage(rotated_plane[half_image_width:2*half_image_width, :])
		right_side = numpy.flip(right_side, 1)
		diff_sides = numpy.square(right_side - left_side)
		count_diff = numpy.count_nonzero(diff_sides)
		if count_diff < min_diff:
			best_angle = angle
			min_diff = count_diff
		angle += step
	rotation = sitk.Euler3DTransform(centroid, 0.0, -best_angle, 0.0)
	moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	composed_transformation.AddTransform(rotation)
	# sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskY.mha"))

	# Align X plane using sagittal heuristics.
	angle = -1.0  # Approximately -60 degrees.
	best_angle = 0.0
	step = 0.01
	image_width = plane_image.GetSize()[0]
	coords = numpy.nonzero(sitk.GetArrayFromImage(moving_image_mask))
	max_gap = max(coords[1])
	rotation_center = (centroid[1], centroid[2])
	while angle <= 1.0:
		rotation2d = sitk.Euler2DTransform((centroid[1], centroid[2]), angle)
		rotated_plane = sitk.Resample(plane_image, plane_image, rotation2d, sitk.sitkNearestNeighbor, 0.0)
		coords = numpy.nonzero(sitk.GetArrayFromImage(rotated_plane))
		gap = max(coords[1])
		if gap > max_gap:
			best_angle = angle
			max_gap = gap
		angle += step
	rotation = sitk.Euler3DTransform(centroid, best_angle, 0.0, 0.0)
	moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	composed_transformation.AddTransform(rotation)
	# sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskX.mha"))
		
    # Resample images after initial alignment.
	moving_image_mask = sitk.Resample(scanBrain,
										atlas_brain_mask,
										composed_transformation,
										sitk.sitkNearestNeighbor,
										0.0,
										sitk.sitkUInt8)

	moving_image = sitk.Resample(scan,
									atlas,
									composed_transformation,
									sitk.sitkBSpline,
									0.0,
									sitk.sitkFloat32)
	
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "AlignedBrainMask.mha"))
	#sitk.WriteImage(moving_image, os.path.join(scanFolder, "AlignedScan.mha"))
	
	return moving_image, moving_image_mask
	

#DATADIR = r'D:/Adam Hilbert/Data/artery_test/Scan.mha'
DATADIR = r"D:\Adam Hilbert\Data\tes\scan.mha"
# ATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/atlas.nii"
ATLAS = "D:/Adam Hilbert/Data/Atlases/MNI_atlas.nii"
# BRAINATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/brain_mask.nii"
BRAINATLAS = "D:/Adam Hilbert/Data/Atlases/brain_mask.nii"
ARTERY = "D:/Adam Hilbert/Data/Atlases/artery_atlas.nii"

validation_root = r'C:\Users\Adam\Registry\ASPECTS_test\Test_12112017'

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--scan', type=str, default=DATADIR,
						help='Path to scan.')
	parser.add_argument('--atlas', type=str, default=ATLAS,
						help='Reference atlas to register the patient\'s brain to.')
	parser.add_argument('--brainatlas', type=str, default=BRAINATLAS,
						help='Brain segment of Reference atlas.')
	
	flags, _ = parser.parse_known_args()

	align(
			scan=flags.scan,
			atlas=flags.atlas,
			atlasBrain=flags.brainatlas,
			cta = False)
	
	#validation_scans = [ [os.path.join(root, name) for name in files if name.endswith(".mha")][0]
	#						for root, dirs, files in os.walk(validation_root) if len(files) == 1]
	
	#for scan in validation_scans:
	#	print(scan.split('\\')[-2] + " running...")
	#	register(
	#		scan=scan,
	#		atlas=flags.atlas,
	#		atlasBrain=flags.brainatlas,
	#		aspects=flags.aspects,
	#		cta = False)
	#	print(scan.split('\\')[-2] + " done.")

