import SimpleITK as sitk
import math
import numpy
import os
from datetime import datetime
import argparse
from multiprocessing.pool import ThreadPool
from functools import partial


def _find_main_axis(image):
	non_zero_elements = numpy.nonzero(image)
	x = non_zero_elements[2]
	y = non_zero_elements[1]
	z = non_zero_elements[0]

	x = x - numpy.mean(x)
	y = y - numpy.mean(y)
	z = z - numpy.mean(z)
	coords = numpy.vstack([x, y, z])

	cov = numpy.cov(coords)
	evals, evecs = numpy.linalg.eig(cov)

	sort_indices = numpy.argsort(evals)[::-1]
	axis1, axis2, axis3 = evecs[:, sort_indices]
	return axis1, axis2, axis3

# This function evaluates the metric value in a thread safe manner
def evaluate_metric(current_rotation, f_image, m_image):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    current_transform = sitk.Euler3DTransform()
    current_transform.SetRotation(*current_rotation)
    registration_method.SetInitialTransform(current_transform)
    res = registration_method.MetricEvaluate(f_image, m_image)
    return res


def MetricEvaluate(fixed_image, modified_moving_image):
	# these parameter value combinations.
	all_orientations = {
						'x=0, y=0, z=90': (0.0, 0.0, numpy.pi / 2.0),
						'x=0, y=0, z=-90': (0.0, 0.0, -numpy.pi / 2.0),
						'x=0, y=0, z=45': (0.0, 0.0, numpy.pi / 4.0),
						'x=0, y=0, z=-45': (0.0, 0.0, -numpy.pi / 4.0),
#						'x=30, y=0, z=0': (numpy.pi / 9.0, 0.0, 0.0),
#						'x=-30, y=0, z=0': (-numpy.pi / 9.0, 0.0, 0.0),
						'x=0, y=45, z=0': (0.0, numpy.pi / 4.0, 0.0),
						'x=0, y=-45, z=0': (0.0, -numpy.pi / 4.0, 0.0),
						'x=0, y=45, z=45': (0.0, numpy.pi / 4.0, numpy.pi / 4.0),
						'x=0, y=45, z=-45': (0.0, numpy.pi / 4.0, -numpy.pi / 4.0),
						'x=0, y=-45, z=45': (0.0, -numpy.pi / 4.0, numpy.pi / 4.0),
						'x=0, y=-45, z=-45': (0.0, -numpy.pi / 4.0, -numpy.pi / 4.0)
						}

	# Evaluate the similarity metric using the eight possible orientations, translation remains the same for all.
	# initial_transform = sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixed_image,
	# 																			modified_moving_image,
	# 																			sitk.Euler3DTransform(),
	# 																			sitk.CenteredTransformInitializerFilter.GEOMETRY))
	# Multi-threaded evaluation
	p = ThreadPool()
	orientations_list = [(0, 0, 0)] + list(all_orientations.values())
	all_metric_values = p.map(partial(evaluate_metric,
									  f_image=fixed_image,
									  m_image=modified_moving_image),
							  		  orientations_list)
	best_orientation = orientations_list[numpy.argmin(all_metric_values)]

	return best_orientation


def align(scan, scanBrain, atlas, atlasBrain, rootTarget, patient, cta=False, epochs=2):

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
	outputspacing = [in_spacing[0] * (in_size[0] / out_size[0]), in_spacing[1] * (in_size[1] / out_size[1]),
					 in_spacing[2] * (in_size[2] / out_size[2])]
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas = resample.Execute(atlas)
	# sitk.WriteImage(atlas, os.path.join(scanFolder, "ResampledAtlas.mha"))

	# Resample atlas_mask to image dimensions
	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(atlas_brain_mask)
	resample.SetSize(out_size)
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	atlas_brain_mask = resample.Execute(atlas_brain_mask)

#	sitk.WriteImage(scanBrain, os.path.join(r"C:\Users\Adam\Registry\NCCT_BL\ST_THINNEST\R0394\ResampledAtlasBrainMask.mha"))

	# Initial transform to align images based on center of mass.
	initial_transform = sitk.CenteredTransformInitializer(atlas_brain_mask,  # fixed image
														  scanBrain,  # moving image
														  sitk.Euler3DTransform(),
														  sitk.CenteredTransformInitializerFilter.MOMENTS)

	# Applying initial transform + correcting for non-square images
	dim = scanBrain.GetSize()[0] if scanBrain.GetSize()[0] < scanBrain.GetSize()[1] else scanBrain.GetSize()[1]
	moving_image_mask = sitk.Resample(scanBrain,  # moving image
									  atlas_brain_mask,  # fixed image
									  initial_transform,
									  sitk.sitkNearestNeighbor,
									  0.0,
									  sitk.sitkUInt8)[
										int(scanBrain.GetSize()[0]/2)-int(dim/2):int(scanBrain.GetSize()[0]/2)+int(dim/2),
										int(scanBrain.GetSize()[1]/2)-int(dim/2):int(scanBrain.GetSize()[1]/2)+int(dim/2), :]

	atlas_brain_mask = atlas_brain_mask[
										int(scanBrain.GetSize()[0]/2)-int(dim/2):int(scanBrain.GetSize()[0]/2)+int(dim/2),
										int(scanBrain.GetSize()[1]/2)-int(dim/2):int(scanBrain.GetSize()[1]/2)+int(dim/2), :]
	scan = scan[
										int(scanBrain.GetSize()[0]/2)-int(dim/2):int(scanBrain.GetSize()[0]/2)+int(dim/2),
										int(scanBrain.GetSize()[1]/2)-int(dim/2):int(scanBrain.GetSize()[1]/2)+int(dim/2), :]
	atlas = atlas[
										int(scanBrain.GetSize()[0]/2)-int(dim/2):int(scanBrain.GetSize()[0]/2)+int(dim/2),
										int(scanBrain.GetSize()[1]/2)-int(dim/2):int(scanBrain.GetSize()[1]/2)+int(dim/2), :]

	composed_transformation = sitk.Transform(3, sitk.sitkComposite)
	composed_transformation.AddTransform(initial_transform)
	sitk.WriteImage(moving_image_mask, os.path.join(rootTarget, patient, "BeforeMetricBrainMask.mha"))

	shape_info = sitk.LabelShapeStatisticsImageFilter()
	shape_info.Execute(moving_image_mask)
	centroid = shape_info.GetCentroid(1)
	centroid_index = sitk.Image.TransformPhysicalPointToContinuousIndex(moving_image_mask, centroid)

	# Check for major mis-orientation
	orient = MetricEvaluate(sitk.Cast(atlas_brain_mask, sitk.sitkFloat32), sitk.Cast(moving_image_mask, sitk.sitkFloat32))

	if not (orient[0] == 0.0 and orient[1] == 0.0 and orient[2] == 0.0):
		rotation = sitk.Euler3DTransform(centroid, orient[0], orient[1], orient[2])
		moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
		composed_transformation.AddTransform(rotation)

#	sitk.WriteImage(moving_image_mask, os.path.join(rootTarget, patient, "BeforeInitialAlignmentBrainMask.mha"))

	for i in range(epochs):
		# Align Z plane using symmetry detection.
		image_width = moving_image_mask.GetSize()[0]
		angle = -(numpy.pi/4.0)-(i*((numpy.pi/4.0)/epochs))
		best_angle = 0.0
		step = 0.05-(i*(0.05/epochs))
		half_image_width = int(image_width / 2)
		min_diff = moving_image_mask.GetSize()[0] * moving_image_mask.GetSize()[1] * moving_image_mask.GetSize()[2] / 2

		while angle <= (numpy.pi/4.0)-(i*((numpy.pi/4.0)/epochs)):
			rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, angle)
			rotated_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
			rotated_mask_data = sitk.GetArrayFromImage(rotated_mask)
			left_side = rotated_mask_data[:, :, 0:half_image_width]
			right_side = rotated_mask_data[:, :, half_image_width:2 * half_image_width]
			right_side = numpy.flip(right_side, 2)
			diff_sides = numpy.square(right_side - left_side)
			count_diff = numpy.sum(diff_sides)
			if count_diff < min_diff:
				best_angle = angle
				min_diff = count_diff
			angle += step

		rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, best_angle)
		moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
		composed_transformation.AddTransform(rotation)
#		print(best_angle)
#		sitk.WriteImage(moving_image_mask, os.path.join(rootTarget, patient, "InitialAlignmentZBrainMask.mha"))

		# Align Y plane using symmetry detection.
		angle = -((numpy.pi/4.0) - (i * ((numpy.pi/4.0) / epochs)))
		best_angle = -((numpy.pi/4.0) - (i * ((numpy.pi/4.0) / epochs)))
		step = 0.05 - (i * (0.05 / epochs))
		half_image_width = int(moving_image_mask.GetSize()[1] / 2)
		min_diff = moving_image_mask.GetSize()[0] * moving_image_mask.GetSize()[1] * moving_image_mask.GetSize()[2] / 2

		while angle <= (numpy.pi/4.0) - (i * ((numpy.pi/4.0) / epochs)):
			rotation = sitk.Euler3DTransform(centroid, 0.0, angle, 0.0)
			rotated_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
			rotated_mask_data = sitk.GetArrayFromImage(rotated_mask)
			left_side = rotated_mask_data[:, :, 0:half_image_width]
			right_side = rotated_mask_data[:, :, half_image_width:2 * half_image_width]
			right_side = numpy.flip(right_side, 2)
			diff_sides = numpy.square(right_side - left_side)
			count_diff = numpy.sum(diff_sides)
			if count_diff < min_diff:
				best_angle = angle
				min_diff = count_diff
			angle += step
		rotation = sitk.Euler3DTransform(centroid, 0.0, best_angle, 0.0)
		moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
		composed_transformation.AddTransform(rotation)
#		print(best_angle)
#		sitk.WriteImage(moving_image_mask, os.path.join(rootTarget, patient, "InitialAlignmentYBrainMask.mha"))

		# Align X plane using sagittal heuristics.
		non_zero_coords = numpy.nonzero(sitk.GetArrayFromImage(moving_image_mask))
		min_y = numpy.argmax(non_zero_coords[1])
		direction = 1.0 if non_zero_coords[0][min_y] < moving_image_mask.GetSize()[2] / 2 else -1.0
		angle = 0.0
		best_angle = 0.0
		step = 0.05 - (i * (0.05 / epochs))
		min_gap = non_zero_coords[1][min_y]
		while abs(angle) <= (numpy.pi/4.0) - (i * ((numpy.pi/4.0) / epochs)):
			rotation = sitk.Euler3DTransform(centroid, angle, 0.0, 0.0)
			rotated_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
			coords = numpy.nonzero(sitk.GetArrayFromImage(rotated_mask))
			gap = max(coords[1])
			if gap > min_gap:
				best_angle = angle
				min_gap = gap
			angle += (direction * step)
		rotation = sitk.Euler3DTransform(centroid, best_angle, 0.0, 0.0)
		moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
		composed_transformation.AddTransform(rotation)
#		print(best_angle)
#		sitk.WriteImage(moving_image_mask, os.path.join(rootTarget, patient, "InitialAlignmentXBrainMask.mha"))

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
								 sitk.sitkLinear,
								 0.0,
								 sitk.sitkFloat32)

	# #sitk.WriteImage(moving_image_mask, os.path.join(scanFolder, "AlignedBrainMask.mha"))
	# #sitk.WriteImage(moving_image, os.path.join(scanFolder, "AlignedScan.mha"))

	return moving_image, moving_image_mask, atlas, atlas_brain_mask


# DATADIR = r'D:/Adam Hilbert/Data/artery_test/Scan.mha'
DATADIR = r"D:\Adam Hilbert\Data\MRCLEAN_REGISTRY\NCCT_BL_ST\R0325\1.0"
# ATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/atlas.nii"
ATLAS = "D:/Adam Hilbert/Data/Atlases/MNI_atlas.nii"
# BRAINATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/brain_mask.nii"
BRAINATLAS = "D:/Adam Hilbert/Data/Atlases/brain_mask.nii"

validation_root = r'C:\Users\Adam\Registry\ASPECTS_test\Test_12112017'

if __name__ == '__main__':
	# Command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--scan', type=str, default="",
						help='Path to scan.')
	parser.add_argument('--atlas', type=str, default="",
						help='Reference atlas to register the patient\'s brain to.')
	parser.add_argument('--brainatlas', type=str, default="",
						help='Brain segment of Reference atlas.')

	flags, _ = parser.parse_known_args()

	ATLAS = os.path.join(flags.assets, "Brainatlas.mha")
	BRAINATLAS = os.path.join(flags.assets, "Brainmask.mha")

	align(
		scan=flags.scan,
		atlas=ATLAS,
		atlasBrain=BRAINATLAS,
		cta=False)

# validation_scans = [ [os.path.join(root, name) for name in files if name.endswith(".mha")][0]
#						for root, dirs, files in os.walk(validation_root) if len(files) == 1]

# for scan in validation_scans:
#	print(scan.split('\\')[-2] + " running...")
#	register(
#		scan=scan,
#		atlas=flags.atlas,
#		atlasBrain=flags.brainatlas,
#		aspects=flags.aspects,
#		cta = False)
#	print(scan.split('\\')[-2] + " done.")
