import numpy as np
import time
import logging
import os.path
import sys
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
from sklearn.grid_search import GridSearchCV


def train(classifier, X_train, y_train):
    #log("Training...")
    print(X_train.shape)
    print(X_train)
    print(y_train.shape)
    print(y_train)
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
    
    return datasetFull

def segment(datasetFull, totalPct, testingPct, randomState):
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
    X = X/255.0*2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testingPct, random_state=randomState)
    
    return(X_train, X_test, y_train, y_test)

def log(message):
    logging.info(message)
    
def score(y_test, y_pred):
    cmtx = confusion_matrix(y_test, y_pred)    
    f1 = f1_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    #roc = roc_auc_score(y_test, y_pred)
    
    return(cmtx, f1, precision, recall)
    
def report(title, configuration, cmtx, f1, precision, recall):
    
    rpt = "\n\nCLASSIFIER: {0}".format(title)
    rpt = rpt + "\n\nConfiguration: {0}".format(configuration)
    
    rpt = rpt + "\n\nConfusion Matrix (vert: actual / horz: predicted) \n\n{0}".format(cmtx)
    
    rpt = rpt + "\n\nScoring Report"
    rpt = rpt + "\n\n{0} \t{1} \t\t{2} \t{3}".format("label", "f1", "precision", "recall")
    for i in range(len(f1)):
        rpt = rpt + "\n{0} \t{1:.5f} \t{2:.5f} \t{3:.5f}".format(i, f1[i], precision[i], recall[i])

    rpt = rpt + "\n{0} \t{1:.5f} \t{2:.5f} \t{3:.5f}\n".format("mean", np.mean(f1), np.mean(precision), np.mean(recall) )

    log("{0}".format(rpt))
    
    return

def execute(X_train, y_train, X_test, y_test):
    
    #parameters = [ { 'n_neighbors': [1,2,3,4,5,6,7,8,9,10] } ]
    #parameters = [ { 'n_estimators': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] } ]
    #parameters = [ { 'kernel': ['rbf'], 'gamma': [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011], 'C': [0.5, 0.75, 1, 2] } ]
    scores = ['f1']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        #clf = GridSearchCV(KNeighborsRegressor(), parameters, cv=5, scoring=score)
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring=score)
        #clf = GridSearchCV(SVC(C=1), parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() / 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        
def OLDexecute(X_train, y_train, X_test, y_test, parameters, scores):

    log("executing classification...")
    
    predictions = []
    f1means = []
    
    # Execute (train, predict, score) each classifier individually
    for i in range(len(parameters)):
        parameter = parameters[i]

        log("Using parameter number: {0}, parameter: {1}".format(i, parameter))
        
        classifier = SVC(**parameter)
        log("Using classifier: {0}".format(classifier))

        train(classifier, X_train, y_train)
        y_pred = predict(classifier, X_test)
        cmtx, f1, precision, recall = score(y_test, y_pred)
        
        predictions.insert(i, y_pred)
        f1mean = np.mean(f1)
        f1means.insert(i, f1mean)

        title = x[0]
        configuration = "{0}".format(classifier)
        report(title, configuration, cmtx, f1, precision, recall)

        log("Completed classifier number: {0}, name: {1}".format(i, title))
        
    return predictions

def visualize(parameters, predictions):
    
    log("predictions: {0}".format(predictions))
    log("parameters: {0}".format(parameters))

    for i in range(parameters):
        parameter = parameterd[i]
        log("parameter: {0}".format(parameter))
        prediction = predictions[i]
        log("prediction: {0}".format(prediction))
        
    
def main():
    
    pctData = 0.01
    if len(sys.argv) > 1:
        pctData = sys.argv[1]
    
    pctTest = 0.3
    if len(sys.argv) > 2:
        pctTest = sys.argv[2]
    
    randomState = 19621015
    if len(sys.argv) > 3:
        randomState = sys.argv[3]
    
    logging.getLogger('').handlers = []
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log("Started mainline")
    log("Using pctData(1): {0}, pctTest(2): {1}, randomState(3): {2}".format(pctData, pctTest, randomState))
    
    trainingFileRaw = "data/train.csv"
    trainingFileNpy = "data/train.npy"   
    dataset = load(trainingFileRaw, trainingFileNpy)
    m, n = dataset.shape

    X_train, X_test, y_train, y_test = segment(dataset, pctData, pctTest, randomState)

    mx_train, nx_train = X_train.shape
    my_train = y_train.shape
    mx_test, nx_test = X_test.shape
    my_test = y_test.shape

    log("Full data set: rows: {0}, features: {1}".format(m,n))    
    log("X training set: rows: {0}, features: {1}".format(mx_train,nx_train))
    log("X cross-validation set: rows: {0}, features: {1}".format(mx_test,nx_test))
    log("y training set: rows: {0}".format(my_train))
    log("y cross-validation set: rows: {0}".format(my_test))
    
    predictions = execute(X_train, y_train, X_test, y_test)
    
    visualize(parameters, predictions)
    
    log("Completed training / cross-validation...")

    log("Starting test data analysis...")

    testFileRaw = "data/test.csv"    
    testFileNpy = "data/test.npy"    
    X = load(testFileRaw, testFileNpy)
    X = X/255.0*2 - 1
    y_pred = predict(classifierSelected, X).astype(int)
    mpred = y_pred.shape
    log("Prediction count: {0}".format(mpred))
    
    # Note: the y_pred needed to be resized to allow CSV
    # See:  http://stackoverflow.com/questions/15454880/delimiter-of-numpy-savetxt
    #predictionsArray = y_pred[None,:]
    
    # Correction -- the data is just a text file with one prediction per row with
    # format as follows (ImageId is sequential number, Label is prediction):
    #    ImageId,Label
    #    1,2
    #    2,0
    #    3,7
    #    4,4
    
    predictionsCSV = "data/predictions-{0}-{1}.csv".format(pctData, pctTest)
    
    predictionsArray = np.zeros( (len(y_pred), 2) ).astype(int)
    for i in range(len(y_pred)):
        prediction = y_pred[i];
        predictionsArray[i][0] = i + 1
        predictionsArray[i][1] = prediction

    print(predictionsArray)
    print(predictionsArray.shape)
    np.savetxt(predictionsCSV, predictionsArray, comments="", header="ImageId,Label", fmt='%s', delimiter=",")
    
    log("Completed prediction, output: {0}".format(predictionsCSV))

    log("Completed test data analysis...")

    log("Completed mainline")
    
if __name__ == "__main__":
    main()    

