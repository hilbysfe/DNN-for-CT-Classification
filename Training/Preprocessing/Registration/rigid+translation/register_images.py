import SimpleITK as sitk
import os
import subprocess


INPUT_FOLDER = "D:\Geert\SOFT_TISSUE_THICK"
OUTPUT_FOLDER = "D:\Geert\SOFT_TISSUE_THICK_REGISTERED_HIGHER_RES"
RESAMPLED_ATLAS_FOLDER = "D:\Geert\RESAMPLED_ATLASSES"
atlas_path = 'atlas.mha'
elastix = sitk.SimpleElastix()
# elastix.LogToConsoleOn()


def resample_atlas(atlas_path, reference_image, output_path):
    # c3d atlas.mha -resample 512x512x30 -o atlas_resampled.mha
    returncode = subprocess.call(['c3d', atlas_path, '-resample', image_size_as_string(reference_image), '-o', output_path])
    resampled_atlas = sitk.Cast( sitk.ReadImage(output_path), sitk.sitkInt16 )
    return (resampled_atlas, returncode)


def image_size_as_string(image):
    sizes = image.GetSize()
    return "x".join([str(size) for size in sizes])


def create_folder_if_not_present(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


create_folder_if_not_present(OUTPUT_FOLDER)
create_folder_if_not_present(RESAMPLED_ATLAS_FOLDER)


elastix.SetParameterMap(sitk.ReadParameterFile("translation.txt"))
elastix.AddParameterMap(sitk.ReadParameterFile("rigid.txt"))
# elastix.AddParameterMap(sitk.ReadParameterFile("affine.txt"))
# elastix.SetParameterMap(sitk.ReadParameterFile("non_rigid.txt"))

atlas = sitk.ReadImage(atlas_path)
already_registered_scans = os.listdir(OUTPUT_FOLDER)

for scan_name in os.listdir(INPUT_FOLDER):
    if not scan_name in already_registered_scans:
        print("Working on {}".format(scan_name))
        relative_path = os.path.join(INPUT_FOLDER, scan_name)
        image = sitk.ReadImage(relative_path)
        image = sitk.Cast(image, sitk.sitkInt16)
        elastix.SetMovingImage(image)
        # Subsample atlas to same dimensions of image. Otherwise, elastix
        # upsamples the image to atlas dimensions before registration, leading to unwanted results.
        # Probably this behaviour can be changed in the parameter file, but I couldn't find anything about it in the docs,
        # so I downsample the atlas manually instead.
        resampled_atlas_path = os.path.join(RESAMPLED_ATLAS_FOLDER, "atlas_" + scan_name)
        resampled_atlas, returncode = resample_atlas(atlas_path, image, resampled_atlas_path)
        print("Resampled atlas. Returncode: {}.".format(returncode))
        elastix.SetFixedImage(resampled_atlas)
        try:
            elastix.Execute()
        except RuntimeError as err:
            print("There was an error in {}.".format(scan_name))
            print(err)
        else:
            result_rigid = elastix.GetResultImage()
            sitk.WriteImage(result_rigid, os.path.join(OUTPUT_FOLDER, scan_name))
