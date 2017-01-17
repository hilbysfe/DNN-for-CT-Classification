
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============ Skull stripping with 2D connected component segmentation ==============

def skullstripping(img_data):

#	rootDir = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\'
#	datasetDir = 'MRCLEAN_CT24h'
#	patient = '\\0001\\CT24h\\thick\\'
#	fileName = 'pat0001.mhd'

#	path = rootDir + datasetDir + patient + fileName

	# --- Read Image ---
#	image = sitk.ReadImage(path)

	# --- Load Data ---

	#center = 30
	#width = 80

	#window_filter = sitk.IntensityWindowingImageFilter()
	#window_filter.SetWindowMinimum(center - (width / 2))
	#window_filter.SetWindowMaximum(center + (width / 2))
	#window_filter.SetOutputMinimum(0)
	#window_filter.SetOutputMaximum(255)
    
#	img_data = sitk.GetArrayFromImage(image)
    
	#img_plot = sitk.GetArrayFromImage(window_filter.Execute(img))
	width = np.shape(img_data)[1]
	height = np.shape(img_data)[2]
	data_new = np.zeros(np.shape(img_data))
	
	for k in range(np.shape(img_data)[0]):
		# --- Thresholding ---
		img = img_data[k,:,:]
		
		for i in range(3,width-3):
			for j in range(3,height-3):
				if img[i,j] < 80 and img[i,j] > 0 \
					and np.min(img[i-3:i+3, j-3:j+3]) < 0 and np.max(img[i-3:i+3, j-3:j+3]) > 80:
					img[i,j] = 0

		g = img < 70
		#print(g)

		# --- Find connected component of brain ---

		w = np.zeros((width,height))

		dx = [0,0,1,1,1,-1,-1,-1]
		dy = [1,-1,0,1,-1,0,1,-1]


		def dfs_it(x, y, c):
			visited, stack = set(), [(x,y)]
			while stack:
				v = stack.pop()
				if v not in visited:
					visited.add(v)
					w[v[0],v[1]] = c
					for i in range(8):
						# for more sophisticated method here could be a check for surrounding average intensity
						nx = v[0]+dx[i]
						ny = v[1]+dy[i]
						if nx >= 0 and ny >= 0 and nx < width and ny < height \
							and g[nx,ny] and (nx,ny) not in visited and img[nx,ny] > 0:
							stack.append((nx,ny))


		s = 1
		for i in range(0,width):
			for j in range(0,height):
				if g[i,j] == True and w[i,j] == 0 and img[i,j] > 10:
					dfs_it(i,j,s)
					s = s+1


	#    print(s)
	#    plt.imshow(w)

		# --- Calculate mean intensity of connected components ---
		mean_int = []
		img_components = []
		component_sizes = []
		for i in range(s):
			ind_i = np.where(w == i)
			component_sizes.append(np.shape(ind_i)[1])
			img_compi = np.zeros((width,height))
			for j in range(np.shape(ind_i)[1]):
				x = ind_i[0][j]
				y = ind_i[1][j]
				img_compi[x][y] = img[x][y]
			mean_int.append(np.true_divide(img_compi.sum(),(img_compi!=0).sum()))
			img_components.append(img_compi)

		# --- Select valid components ---
		valid_means = [i for i in mean_int if i <= 50 and i >= 10]
		valid_indices = [mean_int.index(i) for i in valid_means]
		valid_sizes = [component_sizes[i] for i in valid_indices]
	#    print(valid_means)
	#    print(valid_sizes)

		data_stripped = np.zeros((width,height))
		for i in valid_indices:
			if component_sizes[i] > 5000:
				data_stripped = data_stripped + img_components[i]
				
		data_new[k,:,:] = data_stripped
	#    print(k)
		
#	img_new = sitk.GetImageFromArray(data_new)

	# --- Save image ---
#	newFileName = fileName.split(".")[0] + '_skullstripped.mhd'

#	sitk.WriteImage(img_new, rootDir + datasetDir + patient + newFileName)

	return data_new
