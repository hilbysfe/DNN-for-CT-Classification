from __future__ import division
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import sklearn.svm as svm
from sklearn.metrics import auc
from Utils import utils

def run_svm(dataset_path, svm_model, roc_steps, flag):
	# --------- Load data ----------
	n_classes = 2

	root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
	image_dir = root + dataset_path
	label_filename = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\MRCLEAN\\MRCLEAN_MRSDICH.xlsx'

	dataset = utils.read_dataset(image_dir, label_filename, dense_labels=True)

	fpr = np.zeros((roc_steps))
	tpr = np.zeros((roc_steps))
	tr = np.linspace(0,1,roc_steps)
	acc = 0
	iters = 20

	X_train, labels_train = dataset.Training.next_batch(dataset.Training.num_examples)
	X_val, labels_val = dataset.Validation.next_batch(dataset.Validation.num_examples)

	X_train = X_train.reshape(dataset.Training.num_examples, -1)	
	X_val = X_val.reshape(dataset.Validation.num_examples, -1)
	
	classifier = OneVsRestClassifier(svm_model).fit(X_train, labels_train)
	
	Y_val = classifier.predict(X_val)
	acc = np.sum(Y_val==labels_val)/Y_val.shape[0]		
		
	for k in range(iters):		
		
		X_test, labels_test = dataset.Test.next_batch(dataset.Test.num_examples)
		X_test = X_test.reshape(dataset.Test.num_examples, -1)

		y_score = classifier.predict_proba(X_test)

		# Compute ROC curve and ROC area for each class
		for j in range(roc_steps):
			tp = 0
			fp = 0
			for i in range(np.shape(y_score)[0]):
				if y_score[i][0] >= tr[j]:
					if labels_test[i] == 0:
						tp += 1
					else:
						fp += 1
			tpr[j] += tp/np.sum(labels_test==0) /iters
			fpr[j] += fp/np.sum(labels_test==1) /iters

	roc_auc = auc(fpr, tpr)
	
	np.save('./Statistics/SVM/' + dataset_path + flag + '_tpr.npy', tpr)
	np.save('./Statistics/SVM/' + dataset_path + flag + '_fpr.npy', fpr)
	np.save('./Statistics/SVM/' + dataset_path + flag + '_auc.npy', roc_auc)
	np.save('./Statistics/SVM/' + dataset_path + flag + '_acc.npy', acc)

	print('Multi-class Accuracy and AUC for ' + dataset_path + flag + ' : %s, %s' %(acc,roc_auc))
	
	
# run_svm('Normalized_Resampled_128x128x30', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_RigidAligned_128x128x30', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_Skullstripped_128x128x22', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')

# run_svm('Normalized_Resampled_128x128x30_augmented3', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_RigidAligned_128x128x30_augmented3', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_Skullstripped_128x128x22_augmented3', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')

# run_svm('Normalized_Resampled_128x128x30_augmented5', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_RigidAligned_128x128x30_augmented5', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')
# run_svm('Normalized_Skullstripped_128x128x22_augmented5', svm.SVC(kernel='linear', probability=True, random_state=np.random.RandomState(0)), 200, flag='_linear')


