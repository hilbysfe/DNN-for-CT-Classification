import os


os.system("python train_model.py \
	--learning_rate	0.0005 \
	--max_epochs 10 \
	--batch_size 8 \
	--pretraining False \
	--xvalidation_folds 8 \
	--normalization True \
	--batch_normalization False \
\
	--sigmas 1.5 \
	--kernels 11,3,3 \
	--maps 64,64 \
	--bases 20,20,20 \
	--bases3d True \
\
	--print_freq 1 \
	--eval_freq 1 \
	--checkpoint_freq 0 \
\
	--log_dir /home/nicolab/DATA/logs/0.0005/norm/ \
	--trainingpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/training_points.npy \
	--testpath /home/nicolab/DATA/data_binaries/SUPERVISED/NCCT_THICK_RESAMPLED/test_points.npy"
)


