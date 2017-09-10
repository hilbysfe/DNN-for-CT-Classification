from Utils.utils import split_dataset

training, test = split_dataset('/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_RESAMPLED', 
				'/home/nicolab/DATA/data/Registrydatabase.xlsx',
				'/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/')

print(len(training))
print(len(test))
