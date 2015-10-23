import numpy as np
import time
import logging
import os.path
import sys
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import *


def train(classifier, X_train, y_train):
    log("Training...")
    classifier.fit(X_train, y_train)    
    return(classifier)

def load(pathRaw, pathNpy):
    log("Loading data...")
    
    datasetFull = None    
    
    # Load the data from file and, if necessary, save RAW file as NPY file to speed future use
    if( os.path.exists(pathRaw) ):
        log("Loading NPY dataset: {0}".format(pathNpy) )
        datasetFull = np.load(pathNpy)
    else:
        log("Loading RAW file")
        log("Loading RAW dataset: {0}".format(pathRaw) )
        datasetFull = np.loadtxt(open(pathRaw,"rb"),delimiter=",",skiprows=1)
        log("Saving RAW data into NPY dataset: {0}".format(pathNpy) )
        np.save(pathNpy, datasetFull)    
    return datasetFull

def segment(datasetFull, totalPct, testingPct):
    log("Segmenting data...")

    # Shuffle the data set to randomize the order
    np.random.shuffle(datasetFull)
    
    # Take a small subset of testing hypothesis to lower run-time
    numRows, numFeatures = datasetFull.shape
    rowsUsed = int(numRows * totalPct)
    dataset = datasetFull[0:rowsUsed,:]
    
    # Get metrics on the dataset
    m, n = dataset.shape
    
    y = dataset[:,0]
    X = dataset[:,1:m]
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testingPct, random_state=42)
    
    return(X_train, X_test, y_train, y_test)

def log(message):
    logging.info(message)
    
def RFC():
    log("Using classifier: RFC")
    maxDepth=100
    numEstimators=100
    maxFeatures=100
    classifier = RandomForestClassifier(max_depth=maxDepth, n_estimators=numEstimators, max_features=maxFeatures)
    return(classifier)

def score(classifier, X_train, y_train, X_test, y_test):
    trainingScore = classifier.score(X_train, y_train)
    log("Train Score: {0}".format(trainingScore) )
    
    testScore = classifier.score(X_test, y_test);
    log("Test Score:  {0}".format(testScore) )
    
    y_pred = classifier.predict(X_test)
    
    crpt = classification_report(y_test, y_pred)
    log("\n\nClassification Report \n{0}".format(crpt))

    cmtx = confusion_matrix(y_test, y_pred)    
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    #roc = roc_auc_score(y_test, y_pred)
    
    return(cmtx, f1, precision, recall)
    
def report(cmtx, f1, precision, recall):
    
    log("\n\nConfusion Matrix \n{0}".format(cmtx))
    log("\nf1: {0}".format(f1))
    log("\nprecision: {0}".format(precision))
    log("\nrecall: {0}".format(recall))
    
    rpt = "{0} \t{1} \t\t{2} \t{3}".format("label", "f1", "precision", "recall")
    for i in range(len(f1)):
        rpt = rpt + "\n{0} \t{1:.5f} \t{2:.5f} \t{3:.5f}".format(i, f1[i], precision[i], recall[i])


    log("\n\nScoring Report \n{0}".format(rpt))
    
    return
    

def main():
    
    logging.getLogger('').handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log("Starting mainline")
    
    trainingFileRaw = "data/train.csv"
    trainingFileNpy = "data/train.npy"    
    dataset = load(trainingFileRaw, trainingFileNpy)
    
    X_train, X_test, y_train, y_test = segment(dataset, 0.01, 0.30)
    
    classifier = RFC()
    train(classifier, X_train, y_train)
    cmtx, f1, precision, recall = score(classifier, X_train, y_train, X_test, y_test)
    report(cmtx, f1, precision, recall)
    

if __name__ == "__main__":
    main()

