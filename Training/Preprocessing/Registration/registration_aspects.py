import SimpleITK as sitk
import brain_segmentation as bs
import align
import math
import numpy
import os
from datetime import datetime
import argparse
import time

def register(scanPath, atlasScan, atlasMask, aspects, outputFolder="", cta = False, scanBrain=""):
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

	scan, scan_brain_mask, atlas, atlas_brain_mask = align.align(
		scan=scan,
		scanBrain=scan_brain_mask,
		atlas=atlasScan,
		atlasBrain=atlasMask,
		cta=False)

	sitk.WriteImage(scan_brain_mask, initialAlignmentMask)
	sitk.WriteImage(scan, initialAlignment)

	# Load ASPECTS atlas
	aspects_image = sitk.ReadImage(aspects)

	# Resample aspects to image dimensions
	out_size = scan.GetSize()
	in_size = aspects_image.GetSize()
	in_spacing = aspects_image.GetSpacing()
	outputspacing = [in_spacing[0] * (in_size[0] / out_size[0]), in_spacing[1] * (in_size[1] / out_size[1]),
					 in_spacing[2] * (in_size[2] / out_size[2])]

	resample = sitk.ResampleImageFilter()
	resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
	resample.SetReferenceImage(aspects_image)
	resample.SetSize(out_size)
	resample.SetOutputSpacing(outputspacing)
	resample.SetInterpolator(sitk.sitkNearestNeighbor)
	aspects_image = resample.Execute(aspects_image)	

	#sitk.WriteImage(aspects_image, os.path.join(scanFolder, "ResampledASPECTSMask.mha"))

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
	elastix_filter.SetFixedImage(scan_brain_mask)
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
	parameter_map["MaximumNumberOfIterations"] = ["1000"]
	parameter_map["FinalGridSpacingInPhysicalUnits"] = ["8"]
	parameter_map["GridSpacingSchedule"] = ["3.0", "2.0", "1.5", "1"]
	parameter_map["MaximumStepLength"] = ["12.0", "10.0", "8.0", "6.0"]

	elastix_filter = sitk.SimpleElastix()
	elastix_filter.SetLogToConsole(False)  # True to visualize progress.
	elastix_filter.SetLogToFile(False)
	elastix_filter.SetFixedImage(scan)
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

	
DATADIR = r'C:\Users\Adam\Desktop\R0001\Scan.mha'
ASSETS = r"C:\Users\Adam\Desktop\Assets"

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

#	start = time.time()
	register(
				scanPath=flags.scan,
				atlasScan=ATLAS,
				atlasMask=BRAINATLAS,
				aspects=ASPECTS,
				outputFolder=flags.output,
				cta=False)
#	end = time.time()
#	print(end - start)


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

