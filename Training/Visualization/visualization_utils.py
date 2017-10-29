import numpy as np
from sklearn.feature_extracton.image import extract_patches_2d
from skimage.util.shape import view_as_windows


def zero_padding(input, pSize):
	shape = [ s+2*pSize for s in np.shape(input) ]
	output = np.zeros(shape)
	if len(shape) == 2:
		output[pSize:pSize+np.shape(input)[0], pSize:pSize+np.shape(input)[1]] = input
	elif len(shape) == 3
		output[pSize:pSize+np.shape(input)[0], pSize:pSize+np.shape(input)[1], pSize:pSize+np.shape(input)[2]] = input
	
	return output

def odds(x):
	return np.divide(x, (np.ones(np.shape(x))-x))

def calculateContributionMap(model, input, innerPatchSize, outerPatchSize, numSamples, c):
	# Initialize weight of evidence matrix and counts matrix
	we = np.zeros(np.shape(input))
	counts = np.zeros(np.shape(input))
	
	# Pad input image to define outerpatches
	paddedInput = zero_padding(input, (outerPatchSize-innerPatchSize)/2)
	
	# transform input into list of patches of innerPatchSize
	inputInnerPatches = view_as_windows(input, (innerPatchSize, innerPatchSize))
	inputOuterPatches = view_as_windows(paddedInput, (outerPatchSize, outerPatchSize))
	assert(np.shape(inputInnerPatches)[0] == np.shape(inputOuterPatches)[0] and np.shape(inputInnerPatches)[1] == np.shape(inputOuterPatches)[1])
	
	for i in range(np.shape(inputInnerPatches)[0]):
		for j in range(np.shape(inputInnerPatches)[1]):
			# Define patches
			xw_inner = inputInnerPatches[i,j]
			xw_outer = inputOuterPatches[i,j]
			
			# Copy input + init sumw
			x_copy = input
			sumw = 0
			
			# Sampling
			for s in range(numSamples):
				# Sample from p(xw_inner|xw_outer-xw_inner)
				xw_sampled = sample(xw_inner, xw_outer)
				# Insert sampled patch to copied input
				x_copy[i*innerPatchSize:(i+1)*innerPatchSize, j*innerPatchSize:(j+1)*innerPatchSize] = xw_sampled
				# Evaluate classifier on copied input including sampled patch
				post_xcopy = model(x_copy)
				sumw += post_xcopy[c]
	
			# Calculate p(c|x\xw) and get p(c|x)
			post_xw = sumw / numSamples
			post_x = model(input)
			
			# Update weight of evidence and counts
			we[i*innerPatchSize:(i+1)*innerPatchSize, j*innerPatchSize:(j+1)*innerPatchSize] += (np.log2(odds(post_x)) - np.log2(odds(post_xw)))
			counts[i*innerPatchSize:(i+1)*innerPatchSize, j*innerPatchSize:(j+1)*innerPatchSize] += 1
	
	return np.divide(we, counts)
			
		
			
			
			
			
			
			