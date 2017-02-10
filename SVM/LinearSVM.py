from __future__ import division
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import sklearn.svm as svm

# --------- Load data ----------
root = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\CT24h_Datasets\\'
image_dir = root + 'RigidAligned_256x256x30+Flipped'
label_filename = 'D:\\AdamHilbert\\DNN_Classification_Project\\data\\MRCLEAN\\MRCLEAN_MRSDICH.xlsx'

dataset = read_dataset(image_dir, label_filename)

X_train, labels_train = dataset.Training.next_batch(dataset.Training.num_examples)
X_test, labels_test = dataset.Validation.next_batch(dataset.Validation.num_examples)

X_train = X_train.reshape(dataset.Training.num_examples, -1)
X_test = X_test.reshape(dataset.Validation.num_examples, -1)


Y = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(X_train, labels_train).predict(X_test)

for i in range(2):
	yi = Y[np.where(labels_test==i)[0]]
	acc = np.sum(yi==i)/np.sum(labels_test==i)
	print('Accuracy on class \'%s\': %s' %(i,acc))

acc = np.sum(Y==labels_test)/Y.shape[0]
print('Multi-class Accuracy: %s' %acc)