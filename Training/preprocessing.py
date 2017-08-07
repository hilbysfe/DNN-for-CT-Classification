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

NCCT_MRCLEAN = 			r'E:\MRCLEAN\NCCT_THICK'
NCCT_SKULL_MRCLEAN = 	r'E:\MRCLEAN\NCCT_THICK_SKULLSTRIPPED'
MIP_MRCLEAN = 			r'E:\MRCLEAN\CTA_MIP'
MIP_SKULL_MRCLEAN = 	r'E:\MRCLEAN\CTA_SKULLSTRIPPED_MIP'

NCCT_REGISTRY = 		r'E:\REGISTRY\NCCT_THICK'
NCCT_REGISTRY2 = 		r'E:\NCCT_REGISTRY\NCCT_THICK2'
NCCT_SKULL_REGISTRY = 	r'E:\REGISTRY\NCCT_THICK_SKULLSTRIPPED'
MIP_REGISTRY = 			r'E:\REGISTRY\CTA_MIP'
MIP_SKULL_REGISTRY = 	r'E:\REGISTRY\CTA_SKULLSTRIPPED_MIP'

if __name__ == '__main__':
	


