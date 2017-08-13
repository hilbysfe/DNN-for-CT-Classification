import os

os.system("python train_model.py \
	--learning_rate	0.01 \
	--max_epochs 10 \
	--batch_size 32 \
	--pretraining False \
	--xvalidation_folds 5 \
	--normalization False \
	--batch_normalization False \
\
	--sigmas 1.5 \
	--kernels 11,3,3 \
	--maps 64,64,64 \
	--bases 10,6,6 \
	--bases3d False \
\
	--print_freq 5 \
	--eval_freq 1 \
	--checkpoint_freq 0 \
\
	--log_dir /home/nicolab/DATA/logs/ \
	--trainingpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/training_points.npy \
	--testpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/test_points.npy"
)

	
