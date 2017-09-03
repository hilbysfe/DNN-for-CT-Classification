import numpy as np
import SimpleITK as sitk
import os
from datetime import datetime
import shutil
from multiprocessing import Pool
import scipy.ndimage


def resample(data, image, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([GetSliceThickness(image)] + GetPixelSpacing(image), dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = data.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / data.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(data, real_resize_factor, mode='nearest')
    
    return image, new_spacing



def resample_image(file_in, file_out):
	if not os.path.exists(os.path.dirname(file_out)):
		os.makedirs(os.path.dirname(file_out))
	os.system("c3d %s -resample 512x512x30 -o %s" %(file_in, file_out))
	print(file_in.split('/')[-1] + ' done.')

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
	return len([x for x in list if key in x])


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


def GetSliceThickness(img):
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

def GetPixelSpacing(img):
	return [float(x) for x in img.GetMetaData('0028|0030').split('\\')]


def MoveDirectory(source, target):
	if not os.path.exists(os.path.join(target, source.split('\\')[-1])):
		os.makedirs(os.path.join(target, source.split('\\')[-1]))
		for f in os.listdir(source):
			shutil.move(os.path.join(source, f), os.path.join(target, source.split('\\')[-1], f))
		shutil.rmtree(os.path.abspath(os.path.dirname(source)))


def CopyDirectory(source, target):
	if not os.path.exists(os.path.join(target, source.split('\\')[-1])):
		os.makedirs(os.path.join(target, source.split('\\')[-1]))
		for f in os.listdir(source):
			shutil.copytree(os.path.join(source, f), os.path.join(target, source.split('\\')[-1], f))


def CopyFiles(files, target):
	if not os.path.exists(target):
		os.makedirs(target)
		for f in files:
			shutil.copy(f, target)


def CollectDICOMFolders(path):
	"""
	Returns dictionary with {folder: number of contained .dcm files} syntax.
	"""
	dir_of_all_files = [os.path.abspath(root)
						for root, dirs, files in os.walk(path)
						for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
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
			for f in dicomFolders:
				dicomFiles = [os.path.join(root, name)
							  for root, dirs, files in os.walk(f)
							  for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
				# filter out initializing scans
				blackList = []
				for file in dicomFiles:
					img = sitk.ReadImage(file)
					maxValue = np.power(2, GetBitsStored(img)) / (GetPixelRepresentation(img) + 1) - 1
					if 'LOCALIZER' in GetImageType(img):  # or np.max(sitk.GetArrayFromImage(img)) == maxValue:
						# delete localizer image
						print(file)
						os.remove(file)
						blackList.append(file)

				dicomFiles = [f for f in dicomFiles if f not in blackList]

				# filter out not axial series
				filesSortedX = sort_files(dicomFiles, map=GetXLocation)
				distanceInX = abs(
					GetImagePositionPatient(filesSortedX[1])[0] - GetImagePositionPatient(filesSortedX[-1])[0])
				filesSortedY = sort_files(dicomFiles, map=GetYLocation)
				distanceInY = abs(
					GetImagePositionPatient(filesSortedY[1])[1] - GetImagePositionPatient(filesSortedY[-1])[1])
				filesSortedZ = sort_files(dicomFiles, map=GetZLocation)
				distanceInZ = abs(
					GetImagePositionPatient(filesSortedZ[1])[2] - GetImagePositionPatient(filesSortedZ[-1])[2])

				if distanceInZ < distanceInX or distanceInZ < distanceInY:
					# delete cor / sag series
					shutil.rmtree(f)
					print(f)
					continue
			print(patient + ' done.')
		except:
			print(patient + ' failed.')


def SelectBaseline(rootSource):
	""""
	Selects the baseline series
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
						print(folder)
						shutil.rmtree(folder)
		except:
			print(patient + ' failed.')
			continue

		print(patient + ' done.')


def SelectBySliceThickness(rootSource, rootTarget, thickness):
	""""
	Copies heads only from the thinnest series below the given thickness
	"""
	patients = os.listdir(rootSource)
	for patient in patients:
		if os.path.exists(os.path.join(rootTarget, patient)):
			print(patient + ' already in target.')
			continue
		skip = False
		# if patient == 'R0016':
		try:
			dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
			sliceMap = dict()

			if len(dicomFolders) != 1:
				print(patient + ' more than 1 dicomFolder.')
				continue

			for f in dicomFolders:
				dicomFiles = [os.path.join(root, name)
							  for root, dirs, files in os.walk(f)
							  for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]

				sliceList = [round(GetSliceThickness(file), 1) for file in dicomFiles]

				if len(set(sliceList)) != 1:
					continue
				else:
					if sliceList[0] in sliceMap.keys() and sliceList[0] <= thickness:
						print(patient + ' has multiple series of target thickness.')
						CopyDirectory(os.path.join(rootSource, patient), rootTarget2)
						print(patient + ' copied.')
						skip = True
						break
					sliceMap[sliceList[0]] = f
				if not skip:
					if min(sliceMap.keys()) > thickness:
						print(patient + ' thinnest slice is ' + str(min(sliceMap.keys())))
						continue

				# Copy only head
				headFiles = CropHead(sliceMap[min([t for t in sliceMap.keys() if t <= thickness])])
				CopyFiles(headFiles, os.path.join(rootTarget, patient))
				print(patient + ' done.')
		except:
			print(patient + ' failed.')


def CountSliceThickness(rootSource):
	patients = os.listdir(rootSource)
	sliceMap = dict()
	for patient in patients:
		dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
		slices = []
		for f in dicomFolders:
			dicomFiles = [os.path.join(root, name)
						  for root, dirs, files in os.walk(f)
						  for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]

			sliceList = [round(GetSliceThickness(file), 1) for file in dicomFiles]

			if len(set(sliceList)) != 1:
				print(patient + ' having mixed series.')
				continue

			if sliceList[0] not in slices:
				slices.append(sliceList[0])
		for slice in slices:
			if slice not in sliceMap.keys():
				sliceMap[slice] = 1
			else:
				sliceMap[slice] += 1
		print(patient + ' processed.')
	return sliceMap


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


def CropHeads(rootSource, rootTarget):
	patients = os.listdir(rootSource)
	for patient in patients:
		# if patient == '0001':
		try:
			dicomFolders = CollectDICOMFolders(os.path.join(rootSource, patient))
			for f in dicomFolders:
				dicomFiles = [os.path.join(root, name)
							  for root, dirs, files in os.walk(f)
							  for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
				sortedFiles = sort_files(dicomFiles, map=GetZLocation)[::-1]

				headFiles = []
				top = GetZLocation(sortedFiles[0])
				for file in sortedFiles:
					headFiles.append(file)
					if GetZLocation(file) < top - 200:
						break

				CopyFiles(headFiles, os.path.join(rootTarget, patient))
				print(patient + ' done.')
		except:
			print(patient + ' failed.')


def CropHead(folder):
	dicomFiles = [os.path.join(root, name)
				  for root, dirs, files in os.walk(folder)
				  for name in files if name.endswith((".dcm", ".DCM", ".dicom", ".DICOM"))]
	sortedFiles = sort_files(dicomFiles, map=GetZLocation)[::-1]

	headFiles = []
	top = GetZLocation(sortedFiles[0])
	for file in sortedFiles:
		headFiles.append(file)
		if GetZLocation(file) < top - 200:
			break

	return headFiles

def MIP_DICOM(patient):

	if not os.path.exists(os.path.join(rootTarget, patient)):
		try:
			# Load image
			DicomFolder = os.path.join(rootDicom, patient)
			reader = sitk.ImageSeriesReader()
			series_found = reader.GetGDCMSeriesIDs(DicomFolder)
			if len(series_found) != 1:
				print(patient + ' more series found.')
			filenames = reader.GetGDCMSeriesFileNames(DicomFolder, series_found[0])
			reader.SetFileNames(filenames)
			input_image = reader.Execute()
			input_data = sitk.GetArrayFromImage(input_image)
			sorted_files = sort_files(filenames, map=GetZLocation)

			# Retrieve location information
			sliceLocation = [ GetZLocation(file) for file in sorted_files ]

			# Compute MIP
			mip_slices = []
			i = 0
			offset = 0
			while i < len(sliceLocation):
				gathered = 0.0
				collectedSlices = []
				start = i
				i = offset

				# Collect slice regarding location
				while gathered < WINDOW and i < len(sliceLocation):
					collectedSlices.append(i)
					gathered += abs(sliceLocation[i] - sliceLocation[max(i-1, 0)])
					if gathered < WINDOW - OVERLAP:
						offset = i
					i += 1
				new_slice = np.max(input_data[start:i, :, :], axis=0)
				mip_slices.append(new_slice)

			# Create image
			mip_image = sitk.GetImageFromArray(np.array(mip_slices))

			os.makedirs(os.path.join(rootTarget, patient))
			sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))

			print(patient + ' done.')
		except:
			print(patient + ' failed.')


def MIP_MHA(patient):
	if not os.path.exists(os.path.join(rootTarget, patient)):
		try:
			# Load image
			ImagePath = os.path.join(rootSource, patient, patient + '.mha')
			DicomFolder = os.path.join(rootDicom, patient)
			reader = sitk.ImageSeriesReader()
			series_found = reader.GetGDCMSeriesIDs(DicomFolder)
			if len(series_found) != 1:
				print(patient + ' more series found.')
			filenames = reader.GetGDCMSeriesFileNames(DicomFolder, series_found[0])
			sorted_files = sort_files(filenames, map=GetZLocation)

			input_image = sitk.ReadImage(ImagePath)
			input_data = sitk.GetArrayFromImage(input_image)

			# Retrieve location information
			sliceLocation = [ GetZLocation(file) for file in sorted_files ]

			# Compute MIP
			mip_slices = []
			i = 0
			offset = 0
			while i < len(sliceLocation):
				gathered = 0.0
				collectedSlices = []
				start = i
				i = offset

				# Collect slice regarding location
				while gathered < WINDOW and i < len(sliceLocation):
					collectedSlices.append(i)
					gathered += abs(sliceLocation[i] - sliceLocation[max(i-1, 0)])
					if gathered < WINDOW - OVERLAP:
						offset = i
					i += 1

				new_slice = np.max(input_data[start:i, :, :], axis=0)
				mip_slices.append(new_slice)

			# Create image
			mip_image = sitk.GetImageFromArray(np.array(mip_slices))

			os.makedirs(os.path.join(rootTarget, patient))
			sitk.WriteImage(mip_image, os.path.join(rootTarget, patient, patient + '.mha'))

			print(patient + ' done.')
		except:
			print(patient + ' failed.')

def DICOM2MHA(patient, rootSource, rootTarget):
    # Read image
    DicomFolder = CollectDICOMFolders(os.path.join(rootSource, patient))
    if len(DicomFolder.keys()) != 1:
        print(patient + ' failed.')
        return

    for f in DicomFolder.keys():
        reader = sitk.ImageSeriesReader()
        series_found = reader.GetGDCMSeriesIDs(f)
        if len(series_found) != 1:
            print(patient + ' more series found.')
        filenames = reader.GetGDCMSeriesFileNames(f, series_found[0])
        reader.SetFileNames(filenames)
        input_image = reader.Execute()

        # Write image
        if not os._exists(os.path.join(rootTarget, patient)):
            os.makedirs(os.path.join(rootTarget, patient))
        sitk.WriteImage(input_image, os.path.join(rootTarget, patient, patient + '.mha'))

        print(patient + ' done.')
            
def DICOM2MHA_ALL(rootSource, rootTarget):
	patients = os.listdir(rootSource)
	with Pool() as p:
		p.starmap(DICOM2MHA, zip(patients, np.repeat(rootSource, len(patients)), np.repeat(rootTarget, len(patients))))