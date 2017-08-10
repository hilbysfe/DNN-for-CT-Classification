from Utils.utils import split_dataset

training, test = split_dataset('/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_MIP', 
				'/home/nicolab/DATA/data/Registrydatabase.xlsx',
				'/home/nicolab/DATA/data_binaries/SUPERVISED/CTA_MIP/')

print(len(training))
print(len(test))
