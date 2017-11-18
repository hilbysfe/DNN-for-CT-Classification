import SimpleITK as sitk
import brain_segmentation as bs
import math
import numpy
import os
from datetime import datetime
import argparse
from multiprocessing import Pool

#DATADIR = r'D:/Adam Hilbert/Data/ASPECTS_TestData/Test_12112017/R0001/Scan.mha'
# DATADIR = r'C:/Users/Adam/Registry/NCCT_BL/ST_THIN/R0001/Scan.mha'
# ATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/atlas.nii"
ATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/Brainatlas.mha"
# BRAINATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/brain_mask.nii"
BRAINATLAS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/Brainmask.mha"
ASPECTS = "D:/Adam Hilbert/Data/ASPECTS_TestData/Additional/ASPECTS_areas_original.mhd"

validation_root = r'C:\Users\Adam\Registry\ASPECTS_test\Test_12112017'

def call_registration(scan):
	try:
		print(scan.split('\\')[-2] + " running...")
		os.system("python registration.py \
			--scan " + scan)
	except:
		print(scan.split('\\')[-2] + " failed.")
		return
	print(scan.split('\\')[-2] + " done.")


if __name__ == '__main__':

	validation_scans = [ [os.path.join(root, name) for name in files if name.endswith(".mha")][0]
							for root, dirs, files in os.walk(validation_root) if len(files) > 0]
	
	with Pool() as p:
		p.starmap(call_registration, zip(validation_scans))

	## Command line arguments
	#parser = argparse.ArgumentParser()

	#parser.add_argument('--scan', type=str, default=DATADIR,
	#					help='Path to scan.')
	#parser.add_argument('--atlas', type=str, default=ATLAS,
	#					help='Reference atlas to register the patient\'s brain to.')
	#parser.add_argument('--brainatlas', type=str, default=BRAINATLAS,
	#					help='Brain segment of Reference atlas.')
	#parser.add_argument('--aspects', type=str, default=ASPECTS,
	#					help='Aspects atlas.')

	#flags, _ = parser.parse_known_args()
	
	#register(
	#	scan=flags.scan,
	#	atlas=flags.atlas,
	#	atlasBrain=flags.brainatlas,
	#	aspects=flags.aspects,
	#	cta = False)
