from audioop import avg
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from kneed import KneeLocator



#load dataframe from csv
training = pd.read_csv('resized_dataset.csv')
# training = pd.read_csv('train_dataset.csv')


y = training['Target']
X_raw = training.drop("Target", axis=1)
labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.8)

def cross_validation(classifier):

	#
	# Pass instance of pipeline and training and test data set
	# cv=10 represents the StratifiedKFold with 10 folds
	#
	scores = cross_val_score(classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
	print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def logistic_regression_classification():
# 	Cross Validation accuracy: 0.802 +/- 0.060
# [[365   0   0   0]
#  [  3 386   1   0]
#  [  0   2 378   1]
#  [  0   0   0 392]]
#               precision    recall  f1-score   support

#            0       0.99      1.00      1.00       365
#            1       0.99      0.99      0.99       390
#            2       1.00      0.99      0.99       381
#            3       1.00      1.00      1.00       392

#     accuracy                           1.00      1528
#    macro avg       1.00      1.00      1.00      1528
# weighted avg       1.00      1.00      1.00      1528
	print("Logistic Regression Classifier")
	print("Training set size: ", len(X_train))
	classifier = LogisticRegression(max_iter=100)
	print("Fitting training set...")
	classifier.fit(X_train, y_train)
	print("Predicting...")
	y_predict = classifier.predict(X_test)
	cross_validation(classifier)
	print(classification_report(y_test, y_predict))
	cmd = ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=labels)
	cmd.ax_.set(xlabel="Predicted", ylabel="True")
	# plt.figure(figsize=(8,8))
	coef = np.absolute(classifier.coef_)
	import pixel_mapping
	pixel_mapping.map_pixels(coef, labels)
	# x_axis = [i for i in range(30001)]
	# elbow = []
	# for i in range(0, 4):
	# 	plt.plot(x_axis, sorted(coef[i], reverse=True), label = labels[i])
	# 	kn = KneeLocator(x_axis, sorted(coef[i], reverse=True), curve='convex', direction='decreasing')
	# 	elbow.append(kn.knee)
	# plt.vlines(np.max(elbow), plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	# print('Elbow max: {}'.format(np.max(elbow))) # 19 - 40 depending on random state
	# indexes = [[], [], [], []]
	# for i in range(0, 4):
	# 	important_coefs = sorted(coef[i], reverse=True)[0: np.max(elbow)]
	# 	for c in important_coefs:
	# 		indexes[i].append(list(coef[i]).index(c))
	# plt.legend()
	# plt.show()


def support_vector_classification():
# 	Cross Validation accuracy: 0.793 +/- 0.046
# [[365   0   0   0]
#  [  0 390   0   0]
#  [  0   4 377   0]
#  [  0   0   1 391]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00       365
#            1       0.99      1.00      0.99       390
#            2       1.00      0.99      0.99       381
#            3       1.00      1.00      1.00       392

#     accuracy                           1.00      1528
#    macro avg       1.00      1.00      1.00      1528
# weighted avg       1.00      1.00      1.00      1528
	print("Support Vector Classifier")
	print("Training set size: ", len(X_train))
	classifier = SVC(kernel='linear')
	print("Fitting training set...")
	classifier.fit(X_train, y_train)
	print("Predicting...")
	y_predict = classifier.predict(X_test)
	cross_validation(classifier)
	print(classification_report(y_test, y_predict))
	cmd = ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=labels)
	cmd.ax_.set(xlabel="Predicted", ylabel="True")
	plt.figure(figsize=(8,8))
	plt.show()


def naive_bayes_classification():
# 	Naive Bayes Classifier
# Cross Validation accuracy: 0.586 +/- 0.043
# [[194  43  61  67]
#  [ 34 312  24  20]
#  [ 30 170 153  28]
#  [ 68  40  47 237]]
#               precision    recall  f1-score   support
#            0       0.60      0.53      0.56       365
#            1       0.55      0.80      0.65       390
#            2       0.54      0.40      0.46       381
#            3       0.67      0.60      0.64       392

#     accuracy                           0.59      1528
#    macro avg       0.59      0.58      0.58      1528
# weighted avg       0.59      0.59      0.58      1528
	print("Naive Bayes Classifier")
	# print("Training set size: ", len(X_train))
	nb = MultinomialNB()
	# print("Fitting training set...")
	nb.fit(X_train, y_train)
	# print("Predicting...")
	y_predict = nb.predict(X_test)
	cross_validation(nb)
	print(classification_report(y_test, y_predict))
	cmd = ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=labels)
	cmd.ax_.set(xlabel="Predicted", ylabel="True")
	plt.figure(figsize=(8,8))
	plt.show()


def knn_classification():
	print("KNN Classifier")
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	# print("Predicting...")
	y_predict = knn.predict(X_test)
	cross_validation(knn)
	print(classification_report(y_test, y_predict))
	cmd = ConfusionMatrixDisplay.from_predictions(y_test, y_predict, display_labels=labels)
	cmd.ax_.set(xlabel="Predicted", ylabel="True")
	plt.figure(figsize=(8,8))
	plt.show()

if __name__ == '__main__':
	logistic_regression_classification()
	# support_vector_classification()
	# naive_bayes_classification()
	# knn_classification()