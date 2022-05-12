import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


#load dataframe from csv
training = pd.read_csv('train_dataset.csv')

y = training['Target']
X_raw = training.drop("Target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.6)

def cross_validation(classifier):
	pipeline = make_pipeline(StandardScaler(), classifier)
	#
	# Pass instance of pipeline and training and test data set
	# cv=10 represents the StratifiedKFold with 10 folds
	#
	scores = cross_val_score(pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)
	print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


def logistic_regression_classification():
	print("Logistic Regression Classifier")
	print("Training set size: ", len(X_train))
	classifier = LogisticRegression(max_iter=1000)
	print("Fitting training set...")
	classifier.fit(X_train, y_train)

	print("Predicting...")
	y_predict = classifier.predict(X_test)
	cross_validation(classifier)

def support_vector_classification():
	print("Support Vector Classifier")
	print("Training set size: ", len(X_train))
	classifier = SVC(kernel='linear')
	print("Fitting training set...")
	classifier.fit(X_train, y_train)

	print("Predicting...")
	y_predict = classifier.predict(X_test)

	cross_validation(classifier)

def naive_bayes_classification():
	print("Naive Bayes Classifier")
	# print("Training set size: ", len(X_train))
	nb = GaussianNB()
	# print("Fitting training set...")
	nb.fit(X_train, y_train)
	# print("Predicting...")
	y_predict = nb.predict(X_test)

	cross_validation(nb)

if __name__ == '__main__':
	# logistic_regression_classification()
	# support_vector_classification()
	naive_bayes_classification()