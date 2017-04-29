import os


# CTNET

# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --max_epochs 20 --model_name RFNN_2d --sigmas 1.5,1.0,0.5 \
	# --log_dir ./logs/test/RFNN_128/ --learning_rate 0.0025 " )
	
os.system("python train_cifar.py \
	--dataset_name Normalized_Resampled_256x256x30 --max_epochs 40 --model_name CTNET --sigmas 1.5,1.0,0.5 \
	--log_dir ./logs/test/CTNET_50K/ --learning_rate 0.005 " )