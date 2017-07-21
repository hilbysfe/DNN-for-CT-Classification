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
	if not os.path.exists(os.path.join(target, source.split('\\')[-1])):
		os.makedirs(os.path.join(target, source.split('\\')[-1]))
		for f in os.listdir(source):
			shutil.move(os.path.join(source, f), os.path.join(target, source.split('\\')[-1], f))
		shutil.rmtree(os.path.abspath(os.path.dirname(source)))


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
						print(file)
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
				
				if distanceInZ < distanceInX or distanceInZ < distanceInY:
					# delete cor / sag series
					shutil.rmtree(f)
					print(f)
					continue
			print(patient + ' done.')
		except:
			print(patient + ' failed.')
		
def SelectBaseline(rootSource):
	patients = os.listdir(rootSource)
	for patient in patients:
		try:
			level_1_subdir = GetImmediateSubdirectories(os.path.join(rootSource, patient))[0]
			dicomFolders = CollectDICOMFolders(level_1_subdir)
			dates = dict()			
			for f in dicomFolders:				
				dicomFiles = [os.path.join(root, name)
						 for root, dirs, files in os.walk(f)
						 for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
				sortedFiles = sort_files(dicomFiles, map=GetZLocation)
				
				img = sitk.ReadImage(sortedFiles[0])
				
				d = img.GetMetaData('0008|0020')
				t = img.GetMetaData('0008|0030')[0:6]
				if d is not '' and t is not '':
					date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
				elif d is not '':
					date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:]))

				if date not in dates.keys():
					dates[date] = []
				dates[date].append(f)

			min_date = min(dates.keys())
			for date, folders in dates.items():
				if date != min_date:
					for folder in folders:
						shutil.rmtree(folder)
		except:
			print(patient + ' failed.')
			continue
		
		print(patient + ' done.')

		
def SelectBySliceThickness(rootSource, rootTarget):
	patients = os.listdir(rootSource)
	for patient in patients:
		if os.path.exists(os.path.join(rootTarget, patient)):
			continue
		try:
			level_1_subdir = GetImmediateSubdirectories(os.path.join(rootSource, patient))[0]
			dicomFolders = CollectDICOMFolders(level_1_subdir)
			sliceMap = dict()
			for f in dicomFolders:
				dicomFiles = [os.path.join(root, name)
						 for root, dirs, files in os.walk(f)
						 for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
				
				sliceList = [ GetSliceThickness(file) for file in dicomFiles ]
				
				# if 5.0 in sliceList:
				if len(set(sliceList)) != 1:
					# print(patient + ' failed.')
					continue
				else:
					sliceMap[sliceList[0]] = f
			
			MoveDirectory(sliceMap[max(sliceMap.keys())], rootTarget + '\\' + patient + '\\')			
			print(patient + ' done.')
		except:
			print(patient + ' failed.')

def restore(rootSource, rootTarget):
	patients = os.listdir(rootSource)
	for patient in patients:
		try:
			level_1_subdir = GetImmediateSubdirectories(os.path.join(rootTarget, patient))[0].split('\\')[-1]
			dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
			for f in dicomFolders:			
				path = rootTarget + '\\' + patient + '\\' + level_1_subdir + '\\'
				MoveDirectory(f, path)			
			print(patient + ' done.')
		except:
			print(patient + ' failed.')	
			
def MoveBySlices(rootSource, rootTarget):
	patients = os.listdir(rootSource)
	for patient in patients:
		try:
			level_1_subdir = GetImmediateSubdirectories(os.path.join(rootTarget, patient))[0]
			dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
			
			for f in dicomFolders:
				if dicomFolders[f] > 60:
					MoveDirectory(os.path.join(rootTarget, patient, f), level_1_subdir)
					print(patient + ' moved.')
		except:
			print(patient + ' failed.')

	
# RemoveSagCorLoc('D:\\Adam\\Data\\Registry\\REGISTRY_CTA_BL')
# SelectBaseline('D:\\Adam\\Data\\Registry\\REGISTRY_CTA_BL')

MoveBySlices('D:\\Adam\\Data\\Registry\\5.0\\REGISTRY_NCCT_BL', 'D:\\Adam\\Data\\Registry\\REGISTRY_NCCT_BL')
# restore('D:\\Adam\\Data\\Registry\\5.0\\REGISTRY_CTA_BL', 'D:\\Adam\\Data\\Registry\\REGISTRY_CTA_BL')