'''
Code for training and testing classifier on the trajectory dataset
The data for the code can be found in: https://drive.google.com/drive/folders/1hhFLIcKomL5hH8Aq51aGAVPBDhKw39dF?usp=sharing
test_cls_list.txt, train_cls_list.txt
I use the autoencoder (model3, 64 feature) as a feature extractor, use the features 
from the encoder to fit to a GMM and then sample 1000 features to train classifiers
Needs a gpu to run
Please change the paths before running
To run use: python code_classification_generated.py
'''


import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as TF
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop
from datasets_classifier import TrajectoryDataset, transform
from PIL import Image
from changed_model import *
from torchsummary import summary
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


warnings.filterwarnings("ignore", category=DeprecationWarning)


# CUDA for PyTorch
base_directory = '/path/to/datasets/escape_data/classifier_data'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

## Parameters
batch_size = 16

## Initialize autoencoder model
autoencoder_model = SegModel(training=False)
autoencoder_model.to(device)

## Load saved autoencoder model weights
load_path = '/path/to/model_weights/model_best128_0603.pth.tar'
checkpoint = torch.load(load_path)
epoch = checkpoint['epoch']
train_step = checkpoint['train step']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
print ("------Epoch {} | Step {} | Train Loss {} | Validation Loss {} -----------".format(epoch, train_step, train_loss, val_loss))
autoencoder_model.load_state_dict(checkpoint['state_dict'])
print ("model loaded")

autoencoder_model.eval()
encoder = autoencoder_model.features
encoder.to(device)


def extract_features(img_file, base_directory):
    img_list = [line.rstrip('\n') for line in open(os.path.join(base_directory, img_file))]
    print("--------------------------------------------------------")
    print ("length img_list: ", len(img_list))  ##Debug statements
    print("--------------------------------------------------------")
    data_set = TrajectoryDataset(img_list, transform, splits='test')
    print("--------------------------------------------------------")
    count_data = len(data_set)
    print("Length of dataset: ", count_data)

    ## Initialize data
    features = np.zeros(shape=(count_data, 64))  # torch.Size([1, 64])
    labels = np.zeros(shape=(count_data))  # init_xp = np.zeros(16)

    data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=8)
    i = 0
    for step, testdata in enumerate(data_loader):
        images, label = testdata
        images = images.to(device)
        enc_feat = encoder(images)
        print("encoded feature shape in DATA: ", enc_feat.shape, type(enc_feat))
        print("encoded feature :", enc_feat)
        img_features = enc_feat.cpu()
        img_features = img_features.detach().numpy()
        label = label.cpu()
        label = label.detach().numpy()
        print("label shape as numpy: ", label.shape, type(label))
        print("label numpy :", label)
        print("img_features details:", type(img_features), img_features.shape)
        features[i * batch_size: (i + 1) * batch_size] = img_features
        labels[i * batch_size: (i + 1) * batch_size] = label
        i = i + 1
        if i * batch_size >= count_data:
            break
    print("feature & label size: ", features.shape, labels.shape)

#    print("label[1] feature[1] shapes:", labels[1].shape, features[1].shape)
#    print("label[1] feature[1] type:", type(labels[1]), type(features[1]))
#    print("label[1] :", labels[1], type(labels[1]))
#    print("----------------------------------------------------------------")
#    print("feature[1] :", type(features[1]), features[1])
    return features, labels


if __name__ == '__main__':
	train_dir = 'train_cls_list.txt'
	test_dir = 'test_cls_list.txt'
	train_features, train_labels = extract_features(train_dir, base_directory)
	'''
	Debug statements:
	print("train features: ", type(train_features), train_features.shape)
	print("train labels: ", type(train_labels), train_labels.shape)
	print("train_labels[1]: ", train_labels[1])
	print("----------------------------------------------------------------")
	print("train_features[1]: ", train_features[1])
	print("----------------------------------------------------------------")
	print("train_labels: ", train_labels)
	'''
	print("unique train_labels: ", np.unique(train_labels))
	print("train_features[1]: ", train_features[1])
	
	test_features, test_labels = extract_features(test_dir, base_directory)
	
	# Concatenate training and validation sets. We use all these features for generative modeling with GMM
	
	gen_features = np.concatenate((train_features, test_features))
	gen_labels = np.concatenate((train_labels, test_labels))
	
	print("gen features: ", type(gen_features), gen_features.shape)
	print("gen labels: ", type(gen_labels), gen_labels.shape)
	
	unique, counts = np.unique(gen_labels, return_counts=True)
	print ("unique, counts: ",unique, counts)
	
	total_labels = dict(zip(unique,counts))
	print (total_labels)
	
	key_list = list(total_labels.keys())
	print("keys: ", key_list, type(key_list[0]))
	
	negative_cases = total_labels.get(key_list[0])
	positive_cases = total_labels.get(key_list[1])
	
	print ("No. of positive cases, negative cases: ",positive_cases, negative_cases)
	print ("No. of positive cases, negative cases: ",type(positive_cases), type(negative_cases))
	
	positive_features = np.zeros(shape=(positive_cases, 64))
	negative_features = np.zeros(shape=(negative_cases, 64))
	
	idx_positive = np.where(gen_labels == key_list[1])
	idx_negative = np.where(gen_labels == key_list[0])
	
	print ("Positive indexes: ", len(idx_positive[0]), idx_positive[0].shape, idx_positive[0])
	print ("Negative indexes: ", len(idx_negative[0]), idx_negative[0].shape, idx_negative[0])
	
	for i in range(0,len(idx_positive[0])):
		positive_features[i] = gen_features[idx_positive[0][i]]
	
	print("positive features: ", type(positive_features), positive_features.shape)
	
	for i in range (0, len(idx_negative[0])):
		negative_features[i] = gen_features[idx_negative[0][i]]
	
	print("negative features: ", type(negative_features), negative_features.shape)
	
	
	n_components = np.arange(1, 10)
	models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
	aics = [model.fit(positive_features).aic(positive_features) for model in models]
	bics = [model.fit(positive_features).bic(positive_features) for model in models]
	plt.plot(n_components, aics, label="AIC-positive")
	plt.plot(n_components, bics, label="BIC-positive")
	plt.legend(loc='best')
	plt.xlabel('n_components')
	plt.savefig("/path/to/save/results/positive_data.png")
	print("Plot saved")
	gmm_positive = mixture.GaussianMixture(4, covariance_type='full', random_state=0)
	gmm_positive.fit(positive_features)
	print("Positive GMM: ", gmm_positive.converged_)
	positive_features_new, _ = gmm_positive.sample(n_samples=10000)
	print("positive data from GMM: ", type(positive_features_new), positive_features_new.shape, type(positive_features_new[0]), positive_features_new[0].shape)
	positive_labels_new = np.ones(shape=(10000))
	print("positive labels from GMM: ", type(positive_labels_new), positive_labels_new.shape, type(positive_labels_new[0]), positive_labels_new[0])
	
	
	
	models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
	aics_neg = [model.fit(negative_features).aic(negative_features) for model in models]
	bics_neg = [model.fit(negative_features).bic(negative_features) for model in models]
	plt.plot(n_components, aics_neg, label="AIC-negative")
	plt.plot(n_components, bics_neg, label="BIC-negative")
	plt.legend(loc='best')
	plt.xlabel('n_components')
	plt.savefig("/path/to/save/results//negative_data.png")
	print("Negative Plot saved")
	gmm_negative = mixture.GaussianMixture(4, covariance_type='full', random_state=0)
	gmm_negative.fit(negative_features)
	print("Negative GMM: ", gmm_negative.converged_)
	negative_features_new, _ = gmm_negative.sample(n_samples=10000)
	print("negative data from GMM: ", type(negative_features_new), negative_features_new.shape, type(negative_features_new[0]), negative_features_new[0].shape)
	negative_labels_new = np.zeros(shape=(10000))
	print("negative labels from GMM: ", type(negative_labels_new), negative_labels_new.shape, type(negative_labels_new[0]), negative_labels_new[0])


## Classifier codes start here:	
	scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
	
	svm_features = np.concatenate((positive_features_new, negative_features_new))
	svm_labels = np.concatenate((positive_labels_new, negative_labels_new))
	
	X_train, y_train = svm_features, svm_labels
	X_train, y_train = shuffle(X_train, y_train, random_state=0)

	scaler = MinMaxScaler()
	x_train_minmax = scaler.fit_transform(X_train)

	param = [{
		"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
	}]

	svm = LinearSVC(penalty='l2', loss='hinge', max_iter=10000)
	clf = GridSearchCV(svm, param, cv=10)
	clf.fit(X_train, y_train)
	print("---------best parameters-------")
	print(clf.best_params_)
	score_svm = cross_validate(clf, x_train_minmax, y_train, cv=10, scoring=scoring)
	print("---------SVM scores------------: ", score_svm)
	print("Accuracy: %0.3f (+/- %0.3f)" % (score_svm['test_accuracy'].mean(), score_svm['test_accuracy'].std(ddof=0) * 2))
	print("Precision: %0.3f (+/- %0.3f)" % (score_svm['test_precision'].mean(), score_svm['test_precision'].std(ddof=0) * 2))
	print("Recall: %0.3f (+/- %0.3f)" % (score_svm['test_recall'].mean(), score_svm['test_recall'].std(ddof=0) * 2))
	print("f1_score: %0.3f (+/- %0.3f)" % (score_svm['test_f1_score'].mean(), score_svm['test_f1_score'].std(ddof=0) * 2))

		
	print("-----------Naive bayes-------------")
	gnb = GaussianNB()
	scores = cross_validate(gnb, x_train_minmax, y_train, cv=10, scoring=scoring)
	print('Naive Bayes Score on entire data: ')
	print(scores)
	print("Accuracy: %0.3f (+/- %0.3f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std(ddof=0) * 2))
	print("Precision: %0.3f (+/- %0.3f)" % (scores['test_precision'].mean(), scores['test_precision'].std(ddof=0) * 2))
	print("Recall: %0.3f (+/- %0.3f)" % (scores['test_recall'].mean(), scores['test_recall'].std(ddof=0) * 2))
	print("f1_score: %0.3f (+/- %0.3f)" % (scores['test_f1_score'].mean(), scores['test_f1_score'].std(ddof=0) * 2))


	print("-----------Random Forest-------------")
	rnf = RandomForestClassifier(max_depth=2, random_state=0)
	forest_scores = cross_validate(rnf, x_train_minmax, y_train, cv=10, scoring=scoring)
	print('Random Forest Score on entire data: ')
	print(forest_scores)
	print("Accuracy: %0.3f (+/- %0.3f)" %(forest_scores['test_accuracy'].mean(), forest_scores['test_accuracy'].std(ddof=0) * 2))
	print("Precision: %0.3f (+/- %0.3f)" % (forest_scores['test_precision'].mean(), forest_scores['test_precision'].std(ddof=0) * 2))
	print("Recall: %0.3f (+/- %0.3f)" % (forest_scores['test_recall'].mean(), forest_scores['test_recall'].std(ddof=0) * 2))
	print("f1_score: %0.3f (+/- %0.3f)" % (forest_scores['test_f1_score'].mean(), forest_scores['test_f1_score'].std(ddof=0) * 2))


	print("-----------Decision Tree -------------")
	dt = DecisionTreeClassifier(random_state=0)
	dt_scores = cross_validate(dt, x_train_minmax, y_train, cv=10, scoring=scoring)
	print('Decision Tree Score on entire data: ')
	print(dt_scores)
	print("Accuracy: %0.3f (+/- %0.3f)" % (dt_scores['test_accuracy'].mean(), dt_scores['test_accuracy'].std(ddof=0) * 2))
	print("Precision: %0.3f (+/- %0.3f)" % (dt_scores['test_precision'].mean(), dt_scores['test_precision'].std(ddof=0) * 2))
	print("Recall: %0.3f (+/- %0.3f)" % (dt_scores['test_recall'].mean(), dt_scores['test_recall'].std(ddof=0) * 2))
	print("f1_score: %0.3f (+/- %0.3f)" % (dt_scores['test_f1_score'].mean(), dt_scores['test_f1_score'].std(ddof=0) * 2))


	print("-----------Logistic Regression-------------")
	lr = LogisticRegression(random_state=0)
	lr_scores = cross_validate(lr, x_train_minmax, y_train, cv=10, scoring=scoring)
	print('Logistic regression Score on entire data: ')
	print(lr_scores)
	print("Accuracy: %0.3f (+/- %0.3f)" % (lr_scores['test_accuracy'].mean(), lr_scores['test_accuracy'].std(ddof=0) * 2))
	print("Precision: %0.3f (+/- %0.3f)" % (lr_scores['test_precision'].mean(), lr_scores['test_precision'].std(ddof=0) * 2))
	print("Recall: %0.3f (+/- %0.3f)" % (lr_scores['test_recall'].mean(), lr_scores['test_recall'].std(ddof=0) * 2))
	print("f1_score: %0.3f (+/- %0.3f)" % (lr_scores['test_f1_score'].mean(), lr_scores['test_f1_score'].std(ddof=0) * 2))

	







