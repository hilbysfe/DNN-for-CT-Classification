import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
import shutil
from multiprocessing import Pool
from itertools import repeat

from Preprocessing import preprocessing_utils

WINDOW = 10
OVERLAP = 3

#NCCT_MRCLEAN = 			r'E:\MRCLEAN\NCCT_THICK'
#NCCT_SKULL_MRCLEAN = 		r'E:\MRCLEAN\NCCT_THICK_SKULLSTRIPPED'
#MIP_MRCLEAN = 			r'E:\MRCLEAN\CTA_MIP'
#MIP_SKULL_MRCLEAN = 		r'E:\MRCLEAN\CTA_SKULLSTRIPPED_MIP'

#NCCT_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK'
#NCCT_REGISTRY_RESAMPLED = 	r'E:\REGISTRY\NCCT_THICK_RESAMPLED\'
#NCCT_SKULL_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED'
#NCCT_SKULL_REGISTRY_RESAMPLED =	r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
#MIP_REGISTRY = 			r'E:\REGISTRY\CTA_MIP'
#MIP_SKULL_REGISTRY = 		r'E:\REGISTRY\CTA_SKULLSTRIPPED_MIP'

NCCT_MRCLEAN = 			r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK'
NCCT_SKULL_MRCLEAN = 		r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/NCCT_THICK_SKULLSTRIPPED'
MIP_MRCLEAN = 			r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_MIP'
MIP_SKULL_MRCLEAN = 		r'/home/nicolab/DATA/data/UNSUPERVISED/MRCLEAN/CTA_SKULLSTRIPPED_MIP'

NCCT_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK'
NCCT_REGISTRY_RESAMPLED = 	r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_RESAMPLED'
NCCT_SKULL_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED'
NCCT_SKULL_REGISTRY_RESAMPLED =	r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_SKULLSTRIPPED_RESAMPLED'
MIP_REGISTRY = 			r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_MIP'
MIP_SKULL_REGISTRY = 		r'/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_SKULLSTRIPPED_MIP'


if __name__ == '__main__':
	patients = os.listdir(NCCT_SKULL_REGISTRY)
	inputfiles = [ os.path.join(NCCT_SKULL_REGISTRY, patient, patient + '.mha') for patient in patients]
	outputfiles = [ os.path.join(NCCT_SKULL_REGISTRY_RESAMPLED, patient, patient + '.mha') for patient in patients]
	with Pool() as p:
		p.starmap(preprocessing_utils.resample_image, zip(inputfiles, outputfiles))

