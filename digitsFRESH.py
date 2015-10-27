import numpy as np
import logging
import sys
import os.path

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split

def execute():
    
    #create the training & test sets, skipping the header row with [1:]
    trainingFileRaw = "data/train.csv"
    trainingFileNpy = "data/train.npy"   
    dataset = load(trainingFileRaw, trainingFileNpy)
    m, n = dataset.shape
    
    X_train, X_test, y_train, y_test = segment(dataset, 0.1, 0.30)
    print(X_train.shape)
    print(X_train[0,0:500])
    print(y_train.shape)
    print(y_train[0:100])
    
    target = dataset[0:2940:,0]
    train = dataset[0:2940,1:n]
    print(train.shape)
    print(train[0,0:500])
    print(target.shape)
    print(target[0:100])
    
    classifiers = configure()

    xclassifier = classifiers[0]
    rf = xclassifier[1]
    rf.fit(train, target)
    #rf.fit(X_train, y_train)

    #test = np.load("data/test.npy")
    testFileNpy = "data/test.npy"    
    testFileRaw = "data/test.csv"    
    test = load(testFileRaw, testFileNpy)    
    
    #test = test[0:500,:]

    y_pred= rf.predict(test);
    print(y_pred)
    
    predictionsCSV = "data/predictionsWorking.csv"
    np.savetxt(predictionsCSV, y_pred, fmt="%u", delimiter=",")

def segment(datasetFull, totalPct, testingPct):
    #log("Segmenting data...")

    # Shuffle the data set to randomize the order
    #np.random.shuffle(datasetFull)
    
    # Take a small subset of testing hypothesis to lower run-time
    numRows, numFeatures = datasetFull.shape
    rowsUsed = int(numRows * totalPct)
    dataset = datasetFull[0:rowsUsed,:]
    
    # Get metrics on the dataset
    m, n = dataset.shape
    
    y = dataset[:,0]
    X = dataset[:,1:n]
    
    # Normalize the data
    #X = X/255.0*2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testingPct, random_state=42)
    
    a, b = X_train.shape
    log("X train set: rows: {0}, columns: {1}".format(a,b))

    a, b = X_test.shape
    log("X test set: rows: {0}, columns: {1}".format(a,b))

    a = y_train.shape
    log("y train set: {0}]".format(a))

    a = y_test.shape
    log("y test set: {0}".format(a))

    return(X_train, X_test, y_train, y_test)
    
def configure():
    classifiers = []
    classifiers.append( [ "Random Forest Classifier", RandomForestClassifier(n_estimators=100, n_jobs=2) ] )
    classifiers.append( [ "Nearest Neighbour Regressor", KNeighborsRegressor(n_neighbors=3) ] )
    
    return classifiers

def train(classifier, X_train, y_train):
    #log("Training...")
    classifier.fit(X_train, y_train)    
    return(classifier)
    
def predict(classifier, X):
    y_pred = classifier.predict(X)
    y_pred = np.around(y_pred, decimals=0) 
    return(y_pred)

def load(pathRaw, pathNpy):
    #log("Loading data...")
    
    datasetFull = None    
    
    # Load the data from file and, if necessary, save RAW file as NPY file to speed future use
    if( os.path.exists(pathNpy) ):
        log("Loading NPY dataset: {0}".format(pathNpy) )
        datasetFull = np.load(pathNpy)
    else:
        log("Loading RAW file")
        log("Loading RAW dataset: {0}".format(pathRaw) )
        datasetFull = np.loadtxt(open(pathRaw,"rb"),delimiter=",",skiprows=1)
        log("Saving RAW data into NPY dataset: {0}".format(pathNpy) )
        np.save(pathNpy, datasetFull)    
    
    a, b = datasetFull.shape
    log("Full data set: rows: {0}, columns: {1}".format(a,b))
    
    return datasetFull

def log(message):
    logging.info(message)    

def main():

    logging.getLogger('').handlers = []
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log("Started mainline")
    
    execute()

    log("Completed mainline")

if __name__ == "__main__":
    main()