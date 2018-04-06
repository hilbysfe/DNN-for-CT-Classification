from Utils.utils import split_dataset

training, test = split_dataset(
    '/home/hilbysfe/DATA/SUPERVISED/REGISTRY/NCCT/ST_THINNEST_RESIZED',
    '/home/hilbysfe/DATA/SUPERVISED/REGISTRY/Registrydatabase.xlsx',
    '/home/hilbysfe/DATA/SUPERVISED/REGISTRY/data_binaries/NCCT3D')

print(len(training))
print(len(test))
