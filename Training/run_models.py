import os

os.system("python train_model.py \
	--learning_rate	0.05 \
	--max_epochs 10 \
	--batch_size 32 \
	--pretraining "False" \
	--xvalidation_folds 5 \
\
	--sigmas "1.5" \
	--kernels "11,3,3" \
	--maps "64,64,64" \
\
	--print_freq 1 \
	--eval_freq 2 \
	--checkpoint_freq 0 \
\
	--log_dir "/home/nicolab/DATA/logs/" \
	--trainingpath "/home/nicolab/DATA/data_binaries/training_points.npy" \
	--testpath "/home/nicolab/DATA/data_binaries/test_points.npy" \
	--pretrained_weights_path "" \
	--pretrained_biases_path "" ")

	