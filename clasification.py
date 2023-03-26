import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from kneed import KneeLocator



#load dataframe from csv
training = pd.read_csv('segmented2.csv')
# training = pd.read_csv('train_dataset.csv')


y = training['Target']
X_raw = training.drop("Target", axis=1)
labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, train_size=0.8)

def cross_validation(classifier):
	# Pass instance of pipeline and training and test data set
	# cv=10 represents the StratifiedKFold with 10 folds
	#
	scores = cross_val_score(classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
	print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def logistic_regression_classification():
# Cross Validation accuracy: 0.741 +/- 0.043
#               precision    recall  f1-score   support
#            0       0.76      0.73      0.75       478
#            1       0.80      0.77      0.79       516
#            2       0.73      0.73      0.73       523
#            3       0.73      0.79      0.76       520
#     accuracy                           0.76      2037
#    macro avg       0.76      0.76      0.76      2037
# weighted avg       0.76      0.76      0.76      2037
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
	x_axis = [i for i in range(65536)]
	elbow = []
	for i in range(0, 4):
		plt.plot(x_axis, sorted(coef[i], reverse=True), label = labels[i])
		kn = KneeLocator(x_axis, sorted(coef[i], reverse=True), curve='convex', direction='decreasing')
		elbow.append(kn.knee)
	plt.vlines(np.max(elbow), plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.show()
	print('Elbow max: {}'.format(np.max(elbow))) # 19 - 40 depending on random state
	indexes = [[], [], [], []]
	for i in range(0, 4):
		important_coefs = sorted(coef[i], reverse=True)[0: np.max(elbow)]
		for c in important_coefs:
			indexes[i].append(list(coef[i]).index(c))
	# plt.legend()
	# plt.show()


def support_vector_classification():
# Support Vector Classifier
# Cross Validation accuracy: 0.748 +/- 0.058
#               precision    recall  f1-score   support
#            0       0.70      0.74      0.72       482
#            1       0.79      0.78      0.79       521
#            2       0.79      0.71      0.75       532
#            3       0.70      0.75      0.73       502
#     accuracy                           0.75      2037
#    macro avg       0.75      0.75      0.74      2037
# weighted avg       0.75      0.75      0.75      2037
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
# Cross Validation accuracy: 0.652 +/- 0.048
#               precision    recall  f1-score   support
#            0       0.62      0.64      0.63       479
#            1       0.58      0.76      0.66       511
#            2       0.65      0.56      0.60       523
#            3       0.77      0.63      0.69       524
#     accuracy                           0.65      2037
#    macro avg       0.66      0.65      0.65      2037
# weighted avg       0.66      0.65      0.65      2037
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
# 	Cross Validation accuracy: 0.574 +/- 0.047
#               precision    recall  f1-score   support
#            0       0.52      0.75      0.62       488
#            1       0.63      0.69      0.66       507
#            2       0.48      0.56      0.52       515
#            3       0.78      0.28      0.41       527
#     accuracy                           0.57      2037
#    macro avg       0.61      0.57      0.55      2037
# weighted avg       0.61      0.57      0.55      2037
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