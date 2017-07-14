import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
import shutil

def rotate_images(dataset_path, n):
	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + dataset_path + '\\Training'

	filenames = [filename for filename in os.listdir(image_dir + '\\') if filename.endswith(".mhd")]
	files = [sitk.ReadImage(image_dir + '\\' + filename) for filename in filenames]
	images = np.array([sitk.GetArrayFromImage(file) for file in files])
	print("Loaded " + dataset_path)
	print(images.shape)

	# -90 / 90
	new_image_dir = image_dir + '_augmented' + str(n) + "\\"
	os.mkdir(new_image_dir)

	for i, image in enumerate(images):
		for j in range(1, n + 1):
			angle = np.random.randint(-90, 90)
			new_data = rotate(image, angle, (1, 2))

			w = new_data.shape[1]
			h = new_data.shape[2]

			img = sitk.GetImageFromArray(new_data[:, int(np.floor(w / 2 - 64)):int(np.floor(w / 2 + 64)),
										 int(np.floor(h / 2 - 64)):int(np.floor(h / 2 + 64))])
			img.SetSpacing(files[i].GetSpacing())

			filename = filenames[i].split(".mhd")[0] + '_' + str(j) + '.mhd'

			sitk.WriteImage(img, new_image_dir + filename)
	print("-90/90 done. ")

	# 0 / 360
	new_image_dir = image_dir + '_augmented' + str(n) + "-360\\"
	os.mkdir(new_image_dir)

	for i, image in enumerate(images):
		for j in range(1, n + 1):
			angle = np.random.randint(360)
			new_data = rotate(image, angle, (1, 2))

			w = new_data.shape[1]
			h = new_data.shape[2]

			img = sitk.GetImageFromArray(new_data[:, int(np.floor(w / 2 - 64)):int(np.floor(w / 2 + 64)),
										 int(np.floor(h / 2 - 64)):int(np.floor(h / 2 + 64))])
			img.SetSpacing(files[i].GetSpacing())

			filename = filenames[i].split(".mhd")[0] + '_' + str(j) + '.mhd'

			sitk.WriteImage(img, new_image_dir + filename)
	print("0/360 done. ")


def GetImmediateSubdirectories(a_dir):
	return [os.path.abspath(os.path.join(a_dir, name)) for name in os.listdir(a_dir) if
			os.path.isdir(os.path.join(a_dir, name))]

def count(key, list):
	"""
	Returns how many times key is contained by the elements of list.
	"""
	return len([ x for x in list if key in x])

def sort_files(files, map):
	return sorted(files, key=map)
	
def GetXLocation(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0020|0032').split('\\')[0])
def GetYLocation(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0020|0032').split('\\')[1])
def GetZLocation(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0020|0032').split('\\')[2])
def GetSliceLocation(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0020|1041'))
def GetSliceThickness(file):
	img = sitk.ReadImage(file)
	return float(img.GetMetaData('0018|0050'))	
def GetImagePositionPatient(file):
	img = sitk.ReadImage(file)
	return list(map(lambda x: float(x), img.GetMetaData('0020|0032').split('\\')))
def GetImageOrientationPatient(file):
	img = sitk.ReadImage(file)
	return list(map(lambda x: float(x), img.GetMetaData('0020|0037').split('\\')))

	
def GetStudyDate(img):
	return float(img.GetMetaData('0080|0020'))
def GetStudyTime(img):
	return float(img.GetMetaData('0080|0030'))
def GetImageType(img):
	return img.GetMetaData('0008|0008')
def GetPixelRepresentation(img):
	return int(img.GetMetaData('0028|0103'))
def GetBitsStored(img):
	return int(img.GetMetaData('0028|0101'))
	
	


def MoveDirectory(source, target):
	if not os.path.exists(target):
		os.makedirs(target)
		for f in os.listdir(source):
			shutil.move(os.path.join(source, f), os.path.join(target, f))


def CollectDICOMFolders(path):
	"""
	Returns dictionary with {folder: number of contained .dcm files} syntax.
	"""
	dir_of_all_files = [os.path.abspath(root)
						for root, dirs, files in os.walk(path)
						for name in files
						if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
	return {key: count(key, dir_of_all_files) for key in set(dir_of_all_files)}


def RemoveSagCorLoc(rootSource):
	"""
	Removes all localizer slices from series, then detects and deletes coronal and sagittal series.
	"""
	patients = os.listdir(rootSource)
	lists = []	
	for patient in patients:
		try:
			level_1_subdir = GetImmediateSubdirectories(os.path.join(rootSource, patient))[0]
			dicomFolders = CollectDICOMFolders(level_1_subdir)
			dates = dict()			
			for f in dicomFolders:			
				
				dicomFiles = [os.path.join(root, name)
						 for root, dirs, files in os.walk(f)
						 for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
				
				# filter out initializing scans				
				blackList = []
				for file in dicomFiles:
					img = sitk.ReadImage(file)
					maxValue = np.power(2, GetBitsStored(img))/(GetPixelRepresentation(img)+1) - 1
					if 'LOCALIZER' in GetImageType(img) or np.max(sitk.GetArrayFromImage(img)) == maxValue:
						# delete localizer image
						os.remove(file)						
						blackList.append(file)
				
				dicomFiles = [ f for f in dicomFiles if f not in blackList ]
								
				# filter out not axial series			
				filesSortedX = sort_files(dicomFiles, map=GetXLocation)
				distanceInX = abs(GetImagePositionPatient(filesSortedX[1])[0] - GetImagePositionPatient(filesSortedX[-1])[0])
				filesSortedY = sort_files(dicomFiles, map=GetYLocation)
				distanceInY = abs(GetImagePositionPatient(filesSortedY[1])[1] - GetImagePositionPatient(filesSortedY[-1])[1])
				filesSortedZ = sort_files(dicomFiles, map=GetZLocation)
				distanceInZ = abs(GetImagePositionPatient(filesSortedZ[1])[2] - GetImagePositionPatient(filesSortedZ[-1])[2])
				
				if '1.3.6.1.4.1.40744.9.299291421217849152902168428089734451256' in f:
					print(distanceInX)
					print(distanceInY)
					print(distanceInZ)
				if distanceInZ < distanceInX or distanceInZ < distanceInY:
					# delete cor / sag series
					shutil.rmtree(f)
					print(f)
					continue
			print(patient + ' done.')
		except:
			print(patient + ' failed.')
		
# def SelectBaseline(rootSource, rootTarget):
					
				# try:
					# img = sitk.ReadImage(files[0])
				# except:
					# continue
				# try:
					# d = img.GetMetaData('0008|0020')
					# t = img.GetMetaData('0008|0030')[0:6]
				# except:
					# continue
				# if d is not '' and t is not '':
					# date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
				# elif d is not '':
					# date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:]))

				# if date not in dates.keys():
					# dates[date] = []
				# dates[date].append(f)

			# min_date = min(dates.keys())
			# if len(dates.keys()) > 1:
				# print(patient)
			# for f in dates[min_date]:
				# path = os.path.join(rootTarget, patient, f.split('\\')[-2], f.split('\\')[-1])
				# MoveDirectory(f, path)


RemoveSagCorLoc('D:\\Adam\\Data\\Registry\\REGISTRY_NCCT_BL')
# , 'D:\\Adam Hilbert\\Data\\Registry\\CTA\\')

