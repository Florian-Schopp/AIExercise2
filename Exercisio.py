# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import tree,preprocessing,linear_model
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split, KFold,LeaveOneOut
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt


def exercisio1():
    adult_df = pd.read_csv('class01.csv', header=0, delimiter=' *, *', engine='python')
    adult_df.describe(include='all')
    #declare the first 100 columns as Classifiers
    features = adult_df.values[:, :100]

    # declare the last column as class
    target = adult_df.values[:, 100]

    #split the data in training data(first 350) and testing data
    features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=350, random_state=0)

    #train naive bayes with training set
    clf = GaussianNB()
    clf.fit(features_train, target_train)

    #test the predictor against the train data
    train_pred = clf.predict(features_train)
    print("Accuracy train data")
    print(accuracy_score(target_train, train_pred, normalize=True))

    # test the predictor against the test data
    target_pred = clf.predict(features_test)
    print("Accuracy test data")
    print(accuracy_score(target_test, target_pred, normalize=True))


def exercisio2():
    adult_df = pd.read_csv('class02.csv', header=0, delimiter=' *, *', engine='python')
    adult_df.describe(include='all')
    # declare the first 100 columns as Classifiers
    features = adult_df.values[:, :100]

    # declare the last column as class
    target = adult_df.values[:, 100]
    fold=5
    kf = KFold(n_splits=fold, random_state=None, shuffle=False)
    # iterate over 5 foldings
    avg = 0
    for train_index, test_index in kf.split(features):
        print("Fold: ")
        print(fold)
        features_train = features[train_index]
        features_test = features[test_index]
        target_train = target[train_index]
        target_test = target[test_index]
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(features_train, target_train)
        target_pred = knn.predict(features_test)
        print(accuracy_score(target_test, target_pred, normalize=True))
        avg+=accuracy_score(target_test, target_pred, normalize=True)
    avg /= fold
    print("Accuracy target data")
    print(avg)


def exercisio3():

    adult_df = pd.read_csv('reg01.csv', header=0, delimiter=' *, *', engine='python')
    adult_df.describe(include='all')
    # declare the first 100 columns as Classifiers
    features = adult_df.values[:, :10]

    # declare the last column as class
    target = adult_df.values[:, 10]

    #lab_enc = preprocessing.LabelEncoder()
    #target = lab_enc.fit_transform(target)
    loo = LeaveOneOut()
    rms=0
    k=0
    for train_index, test_index in loo.split(features):
        features_train = features[train_index]
        features_test = features[test_index]
        target_train = target[train_index]
        target_test = target[test_index]
        clf = linear_model.Lasso(alpha=1)
        clf.fit(features_train, target_train)
        pred_test = clf.predict(features_test)
        k =+ 1
        rms+=sqrt(mean_squared_error(pred_test, target_test))

    print(rms/k)


def exercisio4():
    adult_df = pd.read_csv('reg02.csv', header=0, delimiter=' *, *', engine='python')
    adult_df.describe(include='all')
    # declare the first 100 columns as Classifiers
    features = adult_df.values[:, :20]

    # declare the last column as class
    target = adult_df.values[:, 20]

    lab_enc = preprocessing.LabelEncoder()
    target = lab_enc.fit_transform(target)

    # kfold the data with n=5
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    # iterate over 5 foldings
    fold = 1
    MSE = 0
    for train_index, test_index in kf.split(features):
        print("Fold: ")
        print(fold)
        features_train = features[train_index]
        features_test = features[test_index]
        target_train = target[train_index]
        target_test = target[test_index]
        # train deciosion tree with train data
        clf = tree.DecisionTreeClassifier(random_state=0) #max_depth=2 to avoid overfitting? sem realizar podas
        clf = clf.fit(features_train, target_train)

        # test the predictor against the train data
        pred_train = clf.predict(features_train)
        print("MSE for train data")
        print(mean_squared_error(pred_train, target_train))

        # test the predictor against the test data
        pred_test = clf.predict(features_test)
        print("MSE for Test Data")
        MSE += mean_squared_error(pred_test, target_test)
        print(mean_squared_error(pred_test, target_test))
        fold += 1
    print("MSE medio for test data")
    print(MSE/fold)
