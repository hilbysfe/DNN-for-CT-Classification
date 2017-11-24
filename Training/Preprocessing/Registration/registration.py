import SimpleITK as sitk
import brain_segmentation as bs
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

def register(scan, atlas, atlasBrain, artery, cta = True, scanBrain=""):
	scanFolder = os.path.dirname(scan)

	if scanBrain == "":
		scanBrain = os.path.join(scanFolder, "BrainMask.mha")
		if not os.path.exists(scanBrain):
			# Skull stripping part.
			image = sitk.ReadImage(scan)
			if cta:
				image = bs.segment_brain(image, -20, 330, 350)
			else:
				image = bs.segment_brain(image, -20, 140, 160)		
			sitk.WriteImage(image, scanBrain)

	# Load CTA and its brain mask.
	cta = sitk.Cast(sitk.ReadImage(scan), sitk.sitkFloat32)
	cta_brain_mask = sitk.ReadImage(scanBrain)

	# Load atlas and atlas brain mask.
	atlas = sitk.Cast(sitk.ReadImage(atlas), sitk.sitkFloat32)
	atlas_brain_mask = sitk.ReadImage(atlasBrain)

	# Load ASPECTS atlas
	artery_image = sitk.ReadImage(artery)

	# Resample atlas to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(atlas)
	out_size = cta.GetSize()
	in_size = atlas.GetSize()
	resample.SetSize(out_size)
	in_spacing = atlas.GetSpacing()
	outputspacing = [in_spacing[0] * (in_size[0]/out_size[0]), in_spacing[1] * (in_size[1]/out_size[1]), in_spacing[2] * (in_size[2]/out_size[2])]
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas = resample.Execute(atlas)	

	sitk.WriteImage(atlas, os.path.join(scanFolder, "ResampledAtlas.mha"))

	# Resample atlas_mask to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(atlas_brain_mask)
	resample.SetSize(out_size)
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas_brain_mask = resample.Execute(atlas_brain_mask)	

	sitk.WriteImage(atlas_brain_mask, os.path.join(scanFolder, "ResampledAtlasBrainMask.mha"))

	# Resample artery to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(artery_image)
	resample.SetSize(out_size)
	in_size = artery_image.GetSize()
	in_spacing = artery_image.GetSpacing()
	outputspacing = [in_spacing[0] * (in_size[0]/out_size[0]), in_spacing[1] * (in_size[1]/out_size[1]), in_spacing[2] * (in_size[2]/out_size[2])]
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	aspects_image = resample.Execute(artery_image)	

	sitk.WriteImage(artery_image, os.path.join(scanFolder, "ResampledArteryAtlas.mha"))
	
	# Initial transform to align images based on center of mass.
	initial_transform = sitk.CenteredTransformInitializer(atlas_brain_mask,  # fixed image
															cta_brain_mask,  # moving image
															sitk.Euler3DTransform(),
															sitk.CenteredTransformInitializerFilter.MOMENTS)

	moving_image_mask = sitk.Resample(cta_brain_mask,  # moving image
										atlas_brain_mask,  # fixed image
										initial_transform,
										sitk.sitkNearestNeighbor,
										0.0,
										sitk.sitkUInt8)

	#moving_image_mask = cta_brain_mask
	#moving_image = cta

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
		right_side = sitk.GetArrayFromImage(rotated_plane[half_image_width:image_width, :])
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
		right_side = sitk.GetArrayFromImage(rotated_plane[half_image_width:image_width, :])
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
	moving_image_mask = sitk.Resample(cta_brain_mask,
										atlas_brain_mask,
										composed_transformation,
										sitk.sitkNearestNeighbor,
										0.0,
										sitk.sitkUInt8)

	moving_image = sitk.Resample(cta,
									atlas,
									composed_transformation,
									sitk.sitkBSpline,
									0.0,
									sitk.sitkFloat32)

	sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMask.mha"))
	sitk.WriteImage(moving_image, os.path.join(scanFolder, "InitialAlignment.mha"))
	
	# Perform rigid registration.
	#parameter_map = sitk.GetDefaultParameterMap("rigid")
	#parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	#parameter_map["Interpolator"] = ["NearestNeighborInterpolator"]
	#parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	#parameter_map["FixedImagePyramidSchedule"] = ["2", "2", "2", "1", "1", "1"]
	#parameter_map["ImageSampler"] = ["Grid"]
	#parameter_map["SampleGridSpacing"] = ["8"]
	#parameter_map["MaximumStepLength"] = ["8.0", "2.0"]
	#parameter_map["MaximumNumberOfIterations"] = ["100"]
	#parameter_map["MovingImagePyramidSchedule"] = ["2", "2", "2", "1", "1", "1"]
	#parameter_map["NumberOfResolutions"] = ["2"]
	#parameter_map["Metric"] = ["AdvancedMeanSquares"]

	#elastix_filter = sitk.SimpleElastix()
	#elastix_filter.SetLogToConsole(False)  # True to visualize progress.
	#elastix_filter.SetLogToFile(False)
	#elastix_filter.SetFixedImage(moving_image_mask)
	#elastix_filter.SetMovingImage(atlas_brain_mask)
	#elastix_filter.SetParameterMap(parameter_map)
	#elastix_filter.Execute()
	#rigid_result_mask = elastix_filter.GetResultImage()
	#sitk.WriteImage(rigid_result_mask, os.path.join(scanFolder, "RigidRegisteredAtlasMask.mha"))

	#transformix_filter = sitk.SimpleTransformix()
	#transformix_filter.SetMovingImage(atlas)
	#transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	#transformix_filter.SetLogToFile(False)
	#parameter_map = elastix_filter.GetTransformParameterMap()[0]
	#parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
	#parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
	#parameter_map["FixedInternalImagePixelType"] = ["float"]
	#parameter_map["MovingInternalImagePixelType"] = ["float"]
	#parameter_map["ResultImagePixelType"] = ["float"]
	#transformix_filter.SetTransformParameterMap(parameter_map)
	#transformix_filter.Execute()
	#rigid_result_atlas = transformix_filter.GetResultImage()
	#sitk.WriteImage(rigid_result_atlas, os.path.join(scanFolder, "RigidRegisteredAtlas.mha"))

	#transformix_filter = sitk.SimpleTransformix()
	#transformix_filter.SetMovingImage(aspects_image)
	#transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	#transformix_filter.SetLogToFile(False)
	#parameter_map = elastix_filter.GetTransformParameterMap()[0]
	#parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	#parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	#parameter_map["ResultImagePixelType"] = ["short"]
	#transformix_filter.SetTransformParameterMap(parameter_map)
	#transformix_filter.Execute()
	#rigid_result_aspects = transformix_filter.GetResultImage()
	#sitk.WriteImage(rigid_result_aspects, os.path.join(scanFolder, "RigidRegisteredASPECTS.mha"))

    # Perform affine registration.
	parameter_map = sitk.GetDefaultParameterMap("affine")
	parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	parameter_map["Interpolator"] = ["NearestNeighborInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	parameter_map["FixedImagePyramidSchedule"] = ["2", "2", "2", "1", "1", "1"]
	parameter_map["ImageSampler"] = ["Grid"]
	parameter_map["SampleGridSpacing"] = ["8"]
	parameter_map["MaximumStepLength"] = ["8.0", "2.0"]
	parameter_map["MaximumNumberOfIterations"] = ["300"]
	parameter_map["MovingImagePyramidSchedule"] = ["2", "2", "2", "1", "1", "1"]
	parameter_map["NumberOfResolutions"] = ["2"]
	parameter_map["Metric"] = ["AdvancedMeanSquares"]

	elastix_filter = sitk.SimpleElastix()
	elastix_filter.SetLogToConsole(False)  # True to visualize progress.
	elastix_filter.SetLogToFile(False)
	elastix_filter.SetFixedImage(moving_image_mask)
	elastix_filter.SetMovingImage(atlas_brain_mask)
	elastix_filter.SetParameterMap(parameter_map)
	elastix_filter.Execute()
	affine_result_mask = elastix_filter.GetResultImage()
	sitk.WriteImage(affine_result_mask, os.path.join(scanFolder, "AffineRegisteredAtlasMask.mha"))

	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(atlas)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
	parameter_map["FixedInternalImagePixelType"] = ["float"]
	parameter_map["MovingInternalImagePixelType"] = ["float"]
	parameter_map["ResultImagePixelType"] = ["float"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	affine_result_atlas = transformix_filter.GetResultImage()
	sitk.WriteImage(affine_result_atlas, os.path.join(scanFolder, "AffineRegisteredAtlas.mha"))

	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(artery_image)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
	parameter_map["ResultImagePixelType"] = ["float"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	affine_result_artery = transformix_filter.GetResultImage()
	sitk.WriteImage(affine_result_artery, os.path.join(scanFolder, "AffineRegisteredArtery.mha"))

	# Perform non-rigid registration.
	parameter_map = sitk.GetDefaultParameterMap("bspline")
	parameter_map["MaximumNumberOfIterations"] = ["2000"]
	parameter_map["FinalGridSpacingInPhysicalUnits"] = ["8"]
	parameter_map["GridSpacingSchedule"] = ["3.0", "2.0", "1.5", "1"]
	parameter_map["MaximumStepLength"] = ["12.0", "10.0", "8.0", "6.0"]

	elastix_filter = sitk.SimpleElastix()
	elastix_filter.SetLogToConsole(False)  # True to visualize progress.
	elastix_filter.SetLogToFile(False)
	elastix_filter.SetFixedImage(moving_image)
	elastix_filter.SetMovingImage(affine_result_atlas)
	elastix_filter.SetParameterMap(parameter_map)
	elastix_filter.Execute()
	bspline_result_atlas = elastix_filter.GetResultImage()
	sitk.WriteImage(bspline_result_atlas, os.path.join(scanFolder, "BSplineRegisteredAtlas.mha"))

	# Transform brainmask
	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(affine_result_mask)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	parameter_map["ResultImagePixelType"] = ["unsigned char"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	bspline_result_mask = transformix_filter.GetResultImage()
	sitk.WriteImage(bspline_result_mask, os.path.join(scanFolder, "BsplineRegisteredAtlasMask.mha"))

	# Transform aspects
	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(affine_result_artery)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
	parameter_map["ResultImagePixelType"] = ["float"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	bspline_result_artery = transformix_filter.GetResultImage()
	sitk.WriteImage(bspline_result_artery, os.path.join(scanFolder, "BsplineRegisteredArtery.mha"))



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
	parser.add_argument('--artery', type=str, default=ARTERY,
						help='Artery atlas.')

	flags, _ = parser.parse_known_args()

	register(
			scan=flags.scan,
			atlas=flags.atlas,
			atlasBrain=flags.brainatlas,
			artery=flags.artery,
			cta = True)
	
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

