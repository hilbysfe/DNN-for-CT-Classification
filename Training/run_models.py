import os


# CTNET

	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.00015 --batch_size 32 \
	# --kernels 11,3,3 --maps 64,64,64 --max_epochs 60 --l2 0.05")
	

os.system("python train_model.py \
	--dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.0001 --batch_size 64 \
	--kernels 11,11,3,3 --maps 64,64,64,64 --max_epochs 90")	
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.0002 --batch_size 32 \
	# --kernels 11,3,3,3 --maps 64,64,64,64 --max_epochs 70")	
		


	
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --kernels 11,3 --maps 64,64 --max_epochs 60 --cdrop 0.3 --l2 0.15")
	

# CTNET 3D

# os.system("python train_model.py --learning_rate 0.0005 --batch_size 32 --kernels 11")
# os.system("python train_model.py --learning_rate 0.0001 --batch_size 32 --kernels 11")
# os.system("python train_model.py --learning_rate 0.00005 --batch_size 32 --kernels 3")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 32 --kernels 11")
# os.system("python train_model.py --learning_rate 0.001 --batch_size 32 --kernels 11")

# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 5")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 7")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 11")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 11,3")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 11,7")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 11,3,3")
# os.system("python train_model.py --learning_rate 0.00001 --batch_size 16 --kernels 11,7,5")


# Alexnet


# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name alexnet_2d ")
	
	

# Inception

# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30_augmented3 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
	
# os.system("python train_model.py \
	# --dataset_name Normalized_Resampled_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_RigidAligned_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")
# os.system("python train_model.py \
	# --dataset_name Normalized_Skullstripped_128x128x30_augmented5 --learning_rate 0.0001 --batch_size 32 \
	# --max_epochs 60 --model_name Inception ")