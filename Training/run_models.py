import os


# CTNET

os.system("python train_cifar.py \
	--dataset_name Normalized_Resampled_256x256x30 --max_epochs 30 --model_name RFNN_2d --sigmas 1.5,1.0,1.0 ")