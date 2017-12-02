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

def register(scanPath, atlas, atlasBrain, aspects, outputFolder="", cta = True, scanBrain=""):
	if os.path.isfile(scanPath):
		scanFolder = os.path.dirname(scanPath)
		scan = sitk.ReadImage(scanPath)
	else:
		scanFolder = scanPath
		reader = sitk.ImageSeriesReader()
		series_found = reader.GetGDCMSeriesIDs(scanFolder)
		if len(series_found) != 1:
			print('More series found in a single folder.')
			return
		filenames = reader.GetGDCMSeriesFileNames(scanFolder, series_found[0])
		reader.SetFileNames(filenames)
		scan = reader.Execute()

	# --- Define output files ---
	if outputFolder == "":
		initialAlignmentMask = os.path.join(scanFolder, "InitialAlignmentBrainMask.mha")
		initialAlignment = os.path.join(scanFolder, "InitialAlignment.mha")
		bsplineAspects = os.path.join(scanFolder, "BsplineRegisteredASPECTS.mha")
	else:
		if not os.path.exists(outputFolder):
			os.makedirs(outputFolder)
		initialAlignmentMask = os.path.join(outputFolder, "InitialAlignmentBrainMask.mha")
		initialAlignment = os.path.join(outputFolder, "InitialAlignment.mha")
		bsplineAspects = os.path.join(outputFolder, "BsplineRegisteredASPECTS.mha")

	if scanBrain == "":
		if outputFolder == "":
			scanBrain = os.path.join(scanFolder, "ScanBrain.mha")
		else:
			scanBrain = os.path.join(outputFolder, "ScanBrain.mha")
		if not os.path.exists(scanBrain):
			# Skull stripping part.
			if cta:
				scan_brain_mask = bs.segment_brain(scan, -20, 330, 350)
			else:
				scan_brain_mask = bs.segment_brain(scan, -20, 140, 160)		
			sitk.WriteImage(scan_brain_mask, scanBrain)
		else:
			scan_brain_mask = sitk.ReadImage(scanBrain)

	# Load atlas and atlas brain mask.
	atlas = sitk.Cast(sitk.ReadImage(atlas), sitk.sitkFloat32)
	atlas_brain_mask = sitk.ReadImage(atlasBrain)

	# Load ASPECTS atlas
	aspects_image = sitk.ReadImage(aspects)

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

	# Resample aspects to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(aspects_image)
	resample.SetSize(out_size)
	in_size = aspects_image.GetSize()
	in_spacing = aspects_image.GetSpacing()
	outputspacing = [in_spacing[0] * (in_size[0]/out_size[0]), in_spacing[1] * (in_size[1]/out_size[1]), in_spacing[2] * (in_size[2]/out_size[2])]
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	aspects_image = resample.Execute(aspects_image)	

	#sitk.WriteImage(aspects_image, os.path.join(scanFolder, "ResampledASPECTSMask.mha"))
	
	# Initial transform to align images based on center of mass.
	initial_transform = sitk.CenteredTransformInitializer(atlas_brain_mask,  # fixed image
															scan_brain_mask,  # moving image
															sitk.Euler3DTransform(),
															sitk.CenteredTransformInitializerFilter.MOMENTS)

	moving_image_mask = sitk.Resample(scan_brain_mask,  # moving image
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
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "BeforeInitialAlignmentBrainMask.mha"))
	
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

	#reference_plane = int(centroid_index[2])
	reference_plane = int(numpy.argmax(numpy.sum(sitk.GetArrayFromImage(moving_image_mask), (1,2))))
	plane_image = moving_image_mask[:, :, reference_plane]
	angle = -1.3  # Approximately -75 degrees.
	best_angle = 0.0
	step = 0.01
	image_width = moving_image_mask.GetSize()[0]
	half_image_width = int(image_width / 2)
	min_diff = image_width * moving_image_mask.GetSize()[1]
	rotation_center = (centroid[0], centroid[1])
	while angle <= 1.3:
		rotation2d = sitk.Euler2DTransform((centroid[0], centroid[1]), angle)
		rotated_plane = sitk.Resample(plane_image, plane_image, rotation2d, sitk.sitkNearestNeighbor, 0.0)
		left_side = sitk.GetArrayFromImage(rotated_plane[0:half_image_width, :])
		right_side = sitk.GetArrayFromImage(rotated_plane[half_image_width:image_width, :])
		right_side = numpy.flip(right_side, 1)
		diff_sides = numpy.square(right_side - left_side)
		count_diff = numpy.sum(diff_sides)
		if count_diff < min_diff:
			best_angle = angle
			min_diff = count_diff
		angle += step
	rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, best_angle)
	moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	composed_transformation.AddTransform(rotation)
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskZ.mha"))

	# Align Y plane using symmetry detection.
	#reference_plane = int(centroid_index[1])
	reference_plane = int(numpy.argmax(numpy.sum(sitk.GetArrayFromImage(moving_image_mask), (0,1))))
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
		count_diff = numpy.sum(diff_sides)
		if count_diff < min_diff:
			best_angle = angle
			min_diff = count_diff
		angle += step
	rotation = sitk.Euler3DTransform(centroid, 0.0, -best_angle, 0.0)
	moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
	composed_transformation.AddTransform(rotation)
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskY.mha"))

	# Align X plane using sagittal heuristics.
	# reference_plane = int(centroid_index[0])
	# plane_image = moving_image_mask[reference_plane, :, :]
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
	#sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "InitialAlignmentBrainMaskX.mha"))
		

    # Resample images after initial alignment.
	moving_image_mask = sitk.Resample(scan_brain_mask,
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

	sitk.WriteImage(moving_image_mask, initialAlignmentMask)
	sitk.WriteImage(moving_image, initialAlignment)
	

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
	#sitk.WriteImage(affine_result_mask, os.path.join(scanFolder, "AffineRegisteredAtlasMask.mha"))

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
	#sitk.WriteImage(affine_result_atlas, os.path.join(scanFolder, "AffineRegisteredAtlas.mha"))

	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(aspects_image)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	parameter_map["ResultImagePixelType"] = ["short"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	affine_result_aspects = transformix_filter.GetResultImage()
	#sitk.WriteImage(affine_result_aspects, os.path.join(scanFolder, "AffineRegisteredASPECTS.mha"))

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
	#sitk.WriteImage(bspline_result_atlas, os.path.join(scanFolder, "BSplineRegisteredAtlas.mha"))

	# Transform brainmask
	#transformix_filter = sitk.SimpleTransformix()
	#transformix_filter.SetMovingImage(affine_result_mask)
	#transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	#transformix_filter.SetLogToFile(False)
	#parameter_map = elastix_filter.GetTransformParameterMap()[0]
	#parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	#parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	#parameter_map["ResultImagePixelType"] = ["unsigned char"]
	#transformix_filter.SetTransformParameterMap(parameter_map)
	#transformix_filter.Execute()
	#bspline_result_mask = transformix_filter.GetResultImage()
	#sitk.WriteImage(bspline_result_mask, os.path.join(scanFolder, "BsplineRegisteredAtlasMask.mha"))

	# Transform aspects
	transformix_filter = sitk.SimpleTransformix()
	transformix_filter.SetMovingImage(affine_result_aspects)
	transformix_filter.SetLogToConsole(False)  # True to visualize progress.
	transformix_filter.SetLogToFile(False)
	parameter_map = elastix_filter.GetTransformParameterMap()[0]
	parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
	parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
	parameter_map["ResultImagePixelType"] = ["short"]
	transformix_filter.SetTransformParameterMap(parameter_map)
	transformix_filter.Execute()
	bspline_result_aspects = transformix_filter.GetResultImage()
	sitk.WriteImage(bspline_result_aspects, bsplineAspects)

	
DATADIR = r'C:\Users\Adam\Desktop\MrClean0001'
ASSETS = r"D:\Adam Hilbert\Data\Atlases"

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--scan', type=str, default="",
						help='Path to scan. Folder in case of DICOM / filename in case of meta-file format.')
	parser.add_argument('--assets', type=str, default="",
						help='Path to all necessary images.')
	parser.add_argument('--output', type=str, default="",
						help='Output folder results should be saved in.')

	flags, _ = parser.parse_known_args()

	ATLAS = os.path.join(flags.assets, "Brainatlas.mha")
	BRAINATLAS = os.path.join(flags.assets, "Brainmask.mha")
	ASPECTS = os.path.join(flags.assets, "ASPECTS.mha")

	register(
				scanPath=flags.scan,
				atlas=ATLAS,
				atlasBrain=BRAINATLAS,
				aspects=ASPECTS,
				outputFolder = flags.output,
				cta = False)
	


	#validation_scans = [ [os.path.join(root, name) for name in files if name.endswith("Scan.mha")][0]
	#						for root, dirs, files in os.walk(validation_root) if len(files) > 0]
	#print(validation_scans)
	
	#for scan in validation_scans:
	#	print(scan.split('\\')[-2] + " running...")
	#	try:
			#register(
			#	scanPath=flags.scan,
			#	atlas=ATLAS,
			#	atlasBrain=BRAINATLAS,
			#	aspects=ASPECTS,
			#	outputFolder = flags.output,
			#	cta = False)
	#		print(scan.split('\\')[-2] + " done.")
	#	except:
	#		print(scan.split('\\')[-2] + " failed.")
	#		continue

