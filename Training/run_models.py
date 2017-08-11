import os

os.system("python train_model.py \
	--learning_rate	0.05 \
	--max_epochs 10 \
	--batch_size 16 \
	--pretraining False \
	--xvalidation_folds 5 \
	--normalization False \
	--batch_normalization False \
\
	--sigmas 1.5 \
	--kernels 11,3,3 \
	--maps 32,32,32 \
\
	--print_freq 1 \
	--eval_freq 2 \
	--checkpoint_freq 0 \
\
	--log_dir /home/nicolab/DATA/logs/ \
	--trainingpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/training_points.npy \
	--testpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/test_points.npy"
)

	
