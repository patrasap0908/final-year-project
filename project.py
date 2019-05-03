import pandas as pd
import numpy as np
import warnings 
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import keras
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from keras.models import Sequential
from keras.layers import Dense
from subprocess import call



warnings.filterwarnings("ignore")


# Function to under-sample the majority class data
def undersample(data, n=1):
    positive_samples = data[data["Class"] == 1].copy().apply(np.random.permutation)
    negative_samples = data[data["Class"] == 0].copy().apply(np.random.permutation).head(positive_samples.shape[0] * n) 

    positive_samples = data[data["Class"] == 1].copy().apply(np.random.permutation)
    negative_samples = data[data["Class"] == 0].copy().apply(np.random.permutation)[:492]

    undersampled_data = pd.concat([positive_samples, negative_samples])

    return train_test_split(undersampled_data, test_size=0.2)


# Function to normalize the features
def normalize(data): 
    for col in data.columns[:-1]:
        data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())


def visualise(estimator):
	# Export as dot file
	export_graphviz(estimator, out_file='tree.dot', 
	                feature_names = features,
	                class_names = ['0', '1'],
	                rounded = True, proportion = False, 
	                precision = 2, filled = True)

	# Convert to png using system command
	call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

	# Display the image
	img = mpimg.imread('tree.png')
	plt.imshow(img)
	plt.show()


# Different classification models implemented
def RandomForest(X_train, X_test, y_train, y_test, features):
	# Create the classifier model and train it
	clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0).fit(X_train, y_train)

	# Test the trained model
	y_predicted = clf.predict(X_test)

	print(accuracy_score(y_test, y_predicted))
	print()
	print(confusion_matrix(y_test, y_predicted))
	print()
	print(classification_report(y_test, y_predicted))

	# Create a visualization of the trained Random Forest Classifier
	# estimator = clf.estimators_[5]
	# visualise(estimator)


def SVM(X_train, X_test, y_train, y_test):
	# Create the classifier model and train it
	clf = SVC(kernel='linear', gamma='auto').fit(X_train, y_train)

	# Test the trained model
	y_predicted = clf.predict(X_test)

	print(accuracy_score(y_test, y_predicted))
	print()
	print(confusion_matrix(y_test, y_predicted))
	print()
	print(classification_report(y_test, y_predicted))


def ANN(X_train, X_test, y_train, y_test):
	# Scale the data to make the features' values normally distributed
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	# Initialize the ANN
	clf = Sequential()

	# Add two hidden layers and one input layer
	clf.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=29))
	clf.add(Dense(output_dim=6, init='uniform', activation='relu'))
	clf.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

	# Compile the ANN 
	clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# Train the ANN
	clf.fit(X_train, y_train, batch_size=50, nb_epoch=100)

	# Test the trained ANN
	y_predicted = clf.predict(X_test)
	y_predicted = (y_predicted > 0.5)

	print(accuracy_score(y_test, y_predicted))
	print()
	print(confusion_matrix(y_test, y_predicted))
	print()
	print(classification_report(y_test, y_predicted))


def main():
	# Import the data set into a DataFrame object
	data = pd.read_csv("data.csv")

	# Pre-processing and Handle missing values, if any
	data.drop("Time", axis=1, inplace=True)
	data.replace('?', np.nan, inplace=True)
	# print(data.count())
	data.dropna(subset=data.columns, axis=0, inplace=True)

	# print(data.shape)
	X = data
	y = X.pop('Class')
	columns = X.columns
	''' print(X.shape)
	print(y.shape) '''


	# Re-sampling
	X, y = SMOTE(sampling_strategy='minority', random_state=42).fit_resample(X, y)
	''' print(X.shape)
	print(y.shape) '''


	# Splitting the Data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	''' print(X_train.shape)
	print(X_test.shape) '''

	RandomForest(X_train, X_test, y_train, y_test, columns)
	# SVM(X_train[200001:250000], X_test[50001:100000], y_train[200001:250000], y_test[50001:100000])
	# ANN(X_train, X_test, y_train, y_test)



main()