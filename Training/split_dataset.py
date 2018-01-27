from Utils.utils import split_dataset

training, test = split_dataset(
    'D:\Adam Hilbert\Data\Registry\CTA_BL\ST_2D_MIP_RESIZED',
    'E:\MRCLEAN_REGISTRY_BL\Registry_affected_sides.xlsx',
    'D:\Adam Hilbert\Data\data_binaries\MIP2D_affected_side')

print(len(training))
print(len(test))
