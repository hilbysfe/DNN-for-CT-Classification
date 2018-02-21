from Utils.utils import split_dataset

training, test = split_dataset(
    r'D:\Adam Hilbert\Data\Registry\NCCT_BL\ST_THINNEST_RESIZED',
    r'E:\MRCLEAN_REGISTRY_BL\Registrydatabase.xlsx',
    r'D:\Adam Hilbert\Data\data_binaries\NCCT')

print(len(training))
print(len(test))
