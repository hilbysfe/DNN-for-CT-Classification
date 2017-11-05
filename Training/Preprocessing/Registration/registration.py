import SimpleITK as sitk
import brain_segmentation as bs
import math
import numpy


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


# Skull striiping part.
# image = sitk.ReadImage("D:/MRCLEAN/TRIAL/CTA_THIN/0001/Scan.mha")
image = sitk.ReadImage("C:/Users/renan/Downloads/RegTest/Pat9746_1.mha")
image = bs.segment_brain(image, -20, 330, 350)
# sitk.WriteImage(image, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/BrainMask.mha")
sitk.WriteImage(image, "C:/Users/renan/Downloads/RegTest/BrainMask.mha")

# Load CTA and its brain mask.
# cta = sitk.Cast(sitk.ReadImage("D:/MRCLEAN/TRIAL/CTA_THIN/0001/Scan.mha"), sitk.sitkFloat32)
cta = sitk.Cast(sitk.ReadImage("C:/Users/renan/Downloads/RegTest/Pat9746_1.mha"), sitk.sitkFloat32)
cta_brain_mask = sitk.ReadImage("C:/Users/renan/Downloads/RegTest/BrainMask.mha")

# Load atlas and atlas brain mask.
atlas = sitk.ReadImage("D:/MniAtlas/atlas.nii")
atlas_brain_mask = sitk.ReadImage("D:/MniAtlas/brain_mask.nii")

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

composed_transformation = sitk.Transform(3, sitk.sitkComposite)
composed_transformation.AddTransform(initial_transform)

shape_info = sitk.LabelShapeStatisticsImageFilter()
shape_info.Execute(moving_image_mask)
centroid = shape_info.GetCentroid(1)
centroid_index = sitk.Image.TransformPhysicalPointToContinuousIndex(moving_image_mask, centroid)
# sitk.WriteImage(moving_image_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/BeforeInitialAlignmentBrainMask.mha")

# Align Z plane using PCA.
plane_image = moving_image_mask[:, :, int(centroid_index[2])]
axes = _find_main_axis(plane_image)
rotation_info = (axes[0], axes[1]) if abs(axes[0]) > abs(axes[2]) else (axes[2], axes[3])
rotation_direction = rotation_info[0] * rotation_info[1]
rotation_direction = 1.0 if rotation_direction > 0 else -1.0
rotation_angle = rotation_direction * math.acos(abs(rotation_info[0]))
rotation = sitk.Euler3DTransform(centroid, 0.0, 0.0, rotation_angle)
moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
composed_transformation.AddTransform(rotation)
# sitk.WriteImage(moving_image_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/InitialAlignmentBrainMaskZ.mha")

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
# sitk.WriteImage(moving_image_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/InitialAlignmentBrainMaskY.mha")

# Align X plane using PCA.
reference_plane = int(centroid_index[0])
plane_image = moving_image_mask[reference_plane, :, :]
axes = _find_main_axis(plane_image)
rotation_info = (axes[0], axes[1]) if abs(axes[0]) > abs(axes[2]) else (axes[2], axes[3])
rotation_direction = rotation_info[0] * rotation_info[1]
rotation_direction = 1.0 if rotation_direction > 0 else -1.0
rotation_angle = rotation_direction * math.acos(abs(rotation_info[0]))
rotation = sitk.Euler3DTransform(centroid, rotation_angle, 0.0, 0.0)
moving_image_mask = sitk.Resample(moving_image_mask, moving_image_mask, rotation, sitk.sitkNearestNeighbor, 0.0)
composed_transformation.AddTransform(rotation)
# sitk.WriteImage(moving_image_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/InitialAlignmentBrainMaskX.mha")

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

# sitk.WriteImage(moving_image_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/InitialAlignmentBrainMask.mha")
# sitk.WriteImage(moving_image, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/InitialAlignmentScan.mha")

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
elastix_filter.SetFixedImage(atlas_brain_mask)
elastix_filter.SetMovingImage(moving_image_mask)
elastix_filter.SetParameterMap(parameter_map)
elastix_filter.Execute()
affine_result_mask = elastix_filter.GetResultImage()
# sitk.WriteImage(affine_result_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/AffineRegisteredBrainMask.mha")
sitk.WriteImage(affine_result_mask, "C:/Users/renan/Downloads/RegTest/AffineRegisteredBrainMask.mha")

transformix_filter = sitk.SimpleTransformix()
transformix_filter.SetMovingImage(moving_image)
parameter_map = elastix_filter.GetTransformParameterMap()[0]
parameter_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
parameter_map["FinalBSplineInterpolationOrder"] = ["3"]
parameter_map["FixedInternalImagePixelType"] = ["float"]
parameter_map["MovingInternalImagePixelType"] = ["float"]
parameter_map["ResultImagePixelType"] = ["float"]
transformix_filter.SetTransformParameterMap(parameter_map)
transformix_filter.Execute()
affine_result_cta = transformix_filter.GetResultImage()
# sitk.WriteImage(affine_result_cta, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/AffineRegisteredScan.mha")
sitk.WriteImage(affine_result_cta, "C:/Users/renan/Downloads/RegTest/AffineRegisteredScan.mha")

# Perform non-rigid registration.
parameter_map = sitk.GetDefaultParameterMap("bspline")
parameter_map["MaximumNumberOfIterations"] = ["2000"]
parameter_map["FinalGridSpacingInPhysicalUnits"] = ["8"]
parameter_map["GridSpacingSchedule"] = ["3.0", "2.0", "1.5", "1"]
parameter_map["MaximumStepLength"] = ["12.0", "10.0", "8.0", "6.0"]

elastix_filter = sitk.SimpleElastix()
elastix_filter.SetLogToConsole(False)  # True to visualize progress.
elastix_filter.SetLogToFile(False)
elastix_filter.SetFixedImage(atlas)
elastix_filter.SetMovingImage(affine_result_cta)
elastix_filter.SetParameterMap(parameter_map)
elastix_filter.Execute()
bspline_result_cta = elastix_filter.GetResultImage()
# sitk.WriteImage(bspline_result_cta, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/BSplineRegisteredScan.mha")
sitk.WriteImage(bspline_result_cta, "C:/Users/renan/Downloads/RegTest/BSplineRegisteredScan.mha")

transformix_filter = sitk.SimpleTransformix()
transformix_filter.SetMovingImage(affine_result_mask)
parameter_map = elastix_filter.GetTransformParameterMap()[0]
parameter_map["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
parameter_map["ResultImagePixelType"] = ["unsigned char"]
transformix_filter.SetTransformParameterMap(parameter_map)
transformix_filter.Execute()
bspline_result_mask = transformix_filter.GetResultImage()
# sitk.WriteImage(bspline_result_mask, "D:/MRCLEAN/TRIAL/CTA_THIN/0001/BsplineRegisteredBrainMask.mha")
sitk.WriteImage(bspline_result_mask, "C:/Users/renan/Downloads/RegTest/BsplineRegisteredBrainMask.mha")
