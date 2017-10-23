from __future__ import division
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc
from Utils import utils


XVAL_FOLDS = 10
BATCH_SIZE = 128
TRAININGPATH = 
TESTPATH = 

def run_svm(log_path, roc_steps):
	# --------- Load data ----------
	n_classes = 2

	print('Loading Dataset...')
	with open(TRAININGPATH, 'rb') as handle:
		training_points = pickle.load(handle)
	with open(TESTPATH, 'rb') as handle:
		test_points = pickle.load(handle)

	dataset = utils.DataSet(np.array(list(training_points.keys())), np.array(list(training_points.values())),
			np.array(list(test_points.keys())), np.array(list(test_points.values())),
			cross_validation_folds=XVAL_FOLDS,
			normalize = False)
	print('Loading Dataset...done.')
	
	fpr = np.zeros((roc_steps))
	tpr = np.zeros((roc_steps))
	tr = np.linspace(0,1,roc_steps)
	avg_acc = 0
	iters = 5
	max_acc = 0
	
	classifier = SGDClassifier()
	
	for i in range(iters):
		# ------ Repeat xvalidation for robustness ------
		for f in range(XVAL_FOLDS):		
			# -------- TRAIN and VALIDATE in FOLD f --------
			training_steps = int(dataset.Training.num_examples / BATCH_SIZE)
			for i in range(training_steps):
				# ------------ TRAIN -------------
				X_train, labels_train = dataset.Training.next_batch(BATCH_SIZE)

				classifier = classifier.partial_fit(X_train, labels_train)
		
			tot_acc = 0.0
			validation_steps = int(dataset.Validation.num_examples / BATCH_SIZE)
			for step in range(validation_steps):
				# ------------ VALIDATON -------------
				X_val, labels_val = dataset.Validation.next_batch(BATCH_SIZE)
								
				Y_val = classifier.predict(X_val)
				acc = np.sum(Y_val==labels_val)/Y_val.shape[0]
	
				tot_acc += (acc / validation_steps)

			print('Validation accuracy in fold %s: %s' % (f, tot_acc))

			avg_acc += (tot_acc / XVAL_FOLDS)
						
			dataset.next_fold()
		if avg_acc > max_acc:
			max_acc = avg_acc
			best_classifier = classifier

	print('Best Validation accuracy: %s' % max_acc)
	
	for k in range(iters):				
		# ------- TEST -------
		X_test, labels_test = dataset.Test.next_batch(dataset.Test.num_examples)
		X_test = X_test.reshape(dataset.Test.num_examples, -1)

		y_score = best_classifier.predict_proba(X_test)
		Y_test = best_classifier.predict(X_val)
		acc = np.sum(Y_test==labels_test)/Y_test.shape[0]
		
		avg_acc += acc / iters
		
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
	
	np.save(log_path + 'svm_tpr.npy', tpr)
	np.save(log_path + 'svm_fpr.npy', fpr)
	np.save(log_path + 'svm_auc.npy', roc_auc)
	np.save(log_path + 'svm_acc.npy', max_acc)

	print('AUC: %s' % roc_auc)
	

run_svm('/home/nicolab/DATA/logs/Statistics/SVM/', 200)

