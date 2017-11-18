import SimpleITK as sitk
import math


# TODO Remove unnecessary casts.
# TODO Use better naming for intermediate variables.
def segment_brain(image, min_brain_HU, max_brain_HU, min_skull_HU, foramen_radius = 7):
    """
    image          -- input image
    min_brain_HU   -- minimum value of Hounsfield unit that is considered as brain tissue
    max_brain_HU   -- maximum value of Hounsfield unit that is considered as brain tissue
    min_skull_HU   -- bone tissue is everything with this intensity value and above
    foramen_radius -- radius (in mm) of biggest hole in the skull after the foramen magnum
    
    For CTA scans, standard values are the following:
    min_skull_HU = 350
    min_brain_HU = -20
    max_brain_HU = 330

    For non-constrast CT scans, standard values are the following:
    min_skull_HU = 160
    min_brain_HU = -20
    max_brain_HU = 140
    """
    input_image = sitk.Cast(image, sitk.sitkInt32)

    foramen_radius_3d = (math.floor(foramen_radius / input_image.GetSpacing()[0]),
                         math.floor(foramen_radius / input_image.GetSpacing()[1]),
                         math.floor(foramen_radius / input_image.GetSpacing()[2]))

    # Segment all bone in the image.
    aux_image = sitk.BinaryThreshold(input_image, min_skull_HU, 65535)

    # Close skull holes
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetBackgroundValue(0)
    dilate.SetForegroundValue(1)
    dilate.SetKernelRadius(foramen_radius_3d)
    aux_image = dilate.Execute(aux_image)
    skull_without_holes = aux_image

    # TODO Process while loop in parallel.
    # For each scan slice, segment area outside the skull.
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
        aux_image = sitk.Paste(aux_image, slice_mask, slice_mask.GetSize(), destinationIndex = [0, 0, slice_id])
        slice_id = slice_id - 1

    # Dilate the segmentation of the area outside the skull back to the skull border.
    outside_skull_mask = dilate.Execute(aux_image)

    # TODO Continue optimizations from here.
    # Remove other connected components that are not part of the brain.
    outside_brain_mask = outside_skull_mask + skull_without_holes
    outside_brain_mask = sitk.Clamp(outside_brain_mask, lowerBound = 0, upperBound = 1)
    outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)
    outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
    outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)
    outside_brain_mask = sitk.RelabelComponent(sitk.ConnectedComponent(outside_brain_mask))
    outside_brain_mask = sitk.Threshold(outside_brain_mask)

    # Dilate the segmentation of the area inside the skull back to the skull border.
    outside_brain_mask = dilate.Execute(outside_brain_mask)
    outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
    outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)

    # Finds slice with biggest portion of brain.
    slice_id_with_most_brain = 0
    slice_id = input_image.GetSize()[2] - 2  # Because first iteration of while loop is done outside the loop.

    slice_mask = outside_brain_mask[:, :, slice_id]
    slice_mask = sitk.Cast(slice_mask, sitk.sitkInt32)
    label_info_filter = sitk.LabelStatisticsImageFilter()
    label_info_filter.Execute(slice_mask, slice_mask)
    previous_area = label_info_filter.GetCount(0)
    previous_slice_id = slice_id

    while slice_id >= 0:
        slice_mask = outside_brain_mask[:, :, slice_id]
        slice_mask = sitk.Cast(slice_mask, sitk.sitkInt32)
        label_info_filter = sitk.LabelStatisticsImageFilter()
        label_info_filter.Execute(slice_mask, slice_mask)
        area = label_info_filter.GetCount(0)
        if abs(area - previous_area) > 0.05 * area and area < previous_area:
            slice_id_with_most_brain = previous_slice_id
            break
        previous_area = area
        previous_slice_id = slice_id
        slice_id = slice_id - 1

    # HU value of air + threshold to account for noise + value to stay above the max brain HU value.
    hu_delta = 1000 + 100 + (max_brain_HU + abs(min_brain_HU))
    aux_image = outside_brain_mask * hu_delta + sitk.Cast(input_image, sitk.sitkInt32)
    aux_image = sitk.Cast(aux_image, sitk.sitkInt32)

    # Get a seed inside the brain
    slice_with_big_brain_area = outside_brain_mask[:, :, slice_id_with_most_brain]
    slice_with_big_brain_area = sitk.InvertIntensity(slice_with_big_brain_area, 1)
    slice_info = sitk.LabelStatisticsImageFilter()
    slice_info.Execute(slice_with_big_brain_area, slice_with_big_brain_area)
    bound_box = slice_info.GetBoundingBox(1)
    seed_x = int((bound_box[1] - bound_box[0]) / 2 + bound_box[0])
    seed_y = int((bound_box[3] - bound_box[2]) / 2 + bound_box[2])
    seed = (seed_x, seed_y, slice_id_with_most_brain)

    # Use region growing inside the brain
    region_growing_filter = sitk.ConnectedThresholdImageFilter()
    region_growing_filter.SetLower(min_brain_HU)
    region_growing_filter.SetUpper(max_brain_HU)
    region_growing_filter.SetSeed(seed)
    aux_image = region_growing_filter.Execute(aux_image)
    aux_image = sitk.BinaryFillhole(aux_image)
    aux_image = sitk.Cast(aux_image, sitk.sitkUInt8)

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
    air_mask = sitk.Cast(dilate.Execute(sitk.BinaryThreshold(input_image, -32768, -500)), sitk.sitkInt32)

    # Make sure central area of the binary mask has no holes, even when convolution kernel optimized for bone or high
    # intensity structures is used (Kernel H20f)
    core_segmentation = aux_image

    FINAL_KERNEL = (math.floor(4 / input_image.GetSpacing()[0]),
                    math.floor(4 / input_image.GetSpacing()[1]),
                    math.floor(4 / input_image.GetSpacing()[2]))
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetForegroundValue(1)
    dilate.SetBackgroundValue(0)
    dilate.SetKernelRadius(FINAL_KERNEL)

    core_segmentation = dilate.Execute(core_segmentation)

    FINAL_KERNEL = (math.floor(6 / input_image.GetSpacing()[0]),
                    math.floor(6 / input_image.GetSpacing()[1]),
                    math.floor(6 / input_image.GetSpacing()[2]))
    erode = sitk.BinaryErodeImageFilter()
    erode.SetForegroundValue(1)
    erode.SetBackgroundValue(0)
    erode.SetKernelRadius(FINAL_KERNEL)

    core_segmentation = erode.Execute(core_segmentation)
    core_segmentation = sitk.Cast(core_segmentation, sitk.sitkInt32)

    # Segment all bone in the image but now using the brain threshold.
    bone_mask = input_image > max_brain_HU

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
    aux_image = aux_image + core_segmentation
    aux_image = clamp.Execute(aux_image)
    aux_image = sitk.Cast(aux_image, sitk.sitkInt32)
    aux_image = aux_image - air_mask
    aux_image = clamp.Execute(aux_image)
    aux_image = sitk.Cast(aux_image, sitk.sitkInt32)
    aux_image = sitk.RelabelComponent(sitk.ConnectedComponent(aux_image))
    aux_image = sitk.Threshold(aux_image)
    aux_image = sitk.BinaryFillhole(aux_image)

    # TODO Detect neck base based on slice's segmented area connected components (1 component and area < 900mm²)
    neck_slice = 0
    slice_id = slice_id_with_most_brain
    while slice_id >= 0:
        slice_mask = aux_image[:, :, slice_id]
        slice_mask = sitk.Cast(slice_mask, sitk.sitkInt32)
        slice_mask = sitk.ConnectedComponent(slice_mask)
        label_info_filter = sitk.LabelStatisticsImageFilter()
        label_info_filter.Execute(slice_mask, slice_mask)
        area = label_info_filter.GetCount(1) * input_image.GetSpacing()[0] * input_image.GetSpacing()[1]
        components = label_info_filter.GetNumberOfLabels()
        if components == 2 and area < 900:
            neck_slice = slice_id
            break
        slice_id = slice_id - 1

    slice_id = neck_slice - 1
    while slice_id >= 0:
        slice_mask = aux_image[:, :, slice_id]
        slice_mask = sitk.Cast(slice_mask, aux_image.GetPixelID())
        slice_mask = slice_mask * 0
        slice_mask = sitk.JoinSeries(slice_mask)
        aux_image = sitk.Paste(aux_image, slice_mask, slice_mask.GetSize(), destinationIndex = [0, 0, slice_id])
        slice_id = slice_id - 1

    return sitk.Cast(aux_image, sitk.sitkUInt8)
