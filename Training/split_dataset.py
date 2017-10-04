from Utils.utils import split_dataset_NCCT_CTA

training, test = split_dataset_NCCT_CTA(
    '/home/nicolab/DATA/data/SUPERVISED/REGISTRY/NCCT_THICK_RESAMPLED', 
    '/home/nicolab/DATA/data/SUPERVISED/REGISTRY/CTA_THIN_RESAMPLED',
    '/home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_CTA/')

print(len(training))
print(len(test))
