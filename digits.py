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
    
    return datasetFull

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
    X = X/255.0*2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testingPct, random_state=42)
    
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
    
def ensemble(classifiers, X_train, y_train, X_test, y_test):

    log("Ensemble classification...")
    
    predictions = []
    f1means = []
    
    # Execute (train, predict, score) each classifier individually
    for i in range(len(classifiers)):
        x = classifiers[i]
        title = x[0]
        classifier = x[1]

        log("Starting classifier: {0}".format(title))
        
        train(classifier, X_train, y_train)
        y_pred = predict(classifier, X_test)
        cmtx, f1, precision, recall = score(y_test, y_pred)
        
        predictions.insert(i, y_pred)
        f1mean = np.mean(f1)
        f1means.insert(i, f1mean)

        title = x[0]
        configuration = "{0}".format(classifier)
        report(title, configuration, cmtx, f1, precision, recall)

        log("Completed classifier: {0}".format(title))
    
    # Select the best classifer based upon the F1 score
    xf1mean = max(f1means)
    xidx = np.argmax(f1means)
    oclassifier = classifiers[xidx]
    xclassifier = oclassifier[1]
    
    majorityPrediction = vote(predictions, f1means)

    mcmtx, mf1, mprecision, mrecall = score(y_test, majorityPrediction)
    report("Majority Vote", "Majority Vote Configuration", mcmtx, mf1, mprecision, mrecall)

    mf1mean = max(mf1)


    log("Using classifier: {0}".format(xclassifier))
    
    return xclassifier

def vote(predictions, f1means):
    
    #log("predictions: {0}".format(predictions))
    #log("f1means: {0}".format(f1means))

    npPredictions = np.array(predictions)
    numClassifiers, numItems = npPredictions.shape
    #log("INPUT numClassifiers: {0}, numItems: {1}".format(numClassifiers, numItems))
    
    weightedPredictions = []
    j = 0
    k =0
    for i in range(numClassifiers):
        ipredictions = npPredictions[:,i].astype(int)
        #log("i: {0}, ipredictions: {1}".format(i, ipredictions))

        for j in range(len(ipredictions)):
            f1mean = f1means[i]
            #log("f1mean: {0}".format(f1mean))
            if f1mean > 0.97:
                #log("k: {0}".format(k))
                for k in range(k,k+10):
                    #log("(1) i: {0}, j: {1}, k: {2}, adding predictions: {3}".format(i, j, k, predictions[j]))
                    weightedPredictions.insert(k, predictions[j])
                k = k+1
            elif f1mean > 0.95:
                #log("k: {0}".format(k))
                for k in range(k,k+5):
                    #log("(1) i: {0}, j: {1}, k: {2}, adding predictions: {3}".format(i, j, k, predictions[j]))
                    weightedPredictions.insert(k, predictions[j])
                k = k+1
            elif f1mean > 0.90:
                #log("k: {0}".format(k))
                for k in range(k,k+3):
                    #log("(1) i: {0}, j: {1}, k: {2}, adding predictions: {3}".format(i, j, k, predictions[j]))
                    weightedPredictions.insert(k, predictions[j])
                k = k+1
            elif f1mean > 0.85:
                #log("k: {0}".format(k))
                for k in range(k,k+2):
                    #log("(1) i: {0}, j: {1}, k: {2}, adding predictions: {3}".format(i, j, k, predictions[j]))
                    weightedPredictions.insert(k, predictions[j])
                k = k+1
            else:
                #log("(2) i: {0}, j: {1}, k: {2}, adding predictions: {3}".format(i, j, k, predictions[j]))
                weightedPredictions.insert(k, predictions[j])
                k = k+1
        
    # Use majority voting approach
    # note: npPredictions matrix [mxn] 
    #   m = number of classifiers
    #   n = number of items to classify
    npPredictions = np.array(weightedPredictions)
    numClassifiers, numItems = npPredictions.shape
    #log("WEIGHTED numClassifiers: {0}, numItems: {1}".format(numClassifiers, numItems))

    # Get the majority vote for each item
    #   Go through each classifiers by column
    #   Get the majority vote for that classifier
    
    # The majority prediction (majorityPrediction) is [mxn]
    #   m = number of items
    #   n = 1 (one vote)
    majorityPrediction = np.zeros(numItems)
    for i in range(numItems):
        ipredictions = npPredictions[:,i].astype(int)
        vote = np.argmax( np.bincount( ipredictions ) )
        dissentionArray = np.asarray(ipredictions)
        dissentionCount = (dissentionArray != vote).sum()
        dissentionPct = dissentionCount / len(ipredictions)
        if dissentionPct > 0.5:
            log("DISSENTION: element: {0}, count: {1}, vote: {2}, dissention: {3}, dissention PCT: {4:.2f}, ipredictions: {5}".format(i, len(ipredictions), vote, dissentionCount, dissentionPct, ipredictions))
        majorityPrediction[i] = vote

    #log("majorityPrediction: {0}".format(majorityPrediction))
    return majorityPrediction
    
def main():
    
    logging.getLogger('').handlers = []
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log("Started mainline")
    
    log("Start training / cross-validation...")

    trainingFileRaw = "data/train.csv"
    trainingFileNpy = "data/train.npy"   
    dataset = load(trainingFileRaw, trainingFileNpy)
    m, n = dataset.shape

    X_train, X_test, y_train, y_test = segment(dataset, 1.0, 0.30)

    mx_train, nx_train = X_train.shape
    my_train = y_train.shape
    mx_test, nx_test = X_test.shape
    my_test = y_test.shape

    log("Full data set: rows: {0}, features: {1}".format(m,n))    
    log("X training set: rows: {0}, features: {1}".format(mx_train,nx_train))
    log("X cross-validation set: rows: {0}, features: {1}".format(mx_test,nx_test))
    log("y training set: rows: {0}".format(my_train))
    log("y cross-validation set: rows: {0}".format(my_test))
    
    classifiers = []
    classifiers.append( [ "Random Forest Classifier", RandomForestClassifier(n_estimators=100) ] )
    #classifiers.append( [ "Nearest Neighbour Regressor", KNeighborsRegressor(n_neighbors=3) ] )
    #classifiers.append( [ "Decision Tree Classifier", DecisionTreeClassifier(max_depth=10) ] )
    classifiers.append( [ "AdaBoost Classifier", AdaBoostClassifier(DecisionTreeClassifier(max_depth=10)) ] )
    #classifiers.append( [ "LDA Classifier", LDA() ] )
    #classifiers.append( [ "Guassian NB Classifier", GaussianNB() ] )
    #classifiers.append( [ "QDA Classifier", QDA(priors=None, reg_param=0.5) ] )
    classifiers.append( [ "Logistic Regression", LogisticRegression() ] )
    #classifiers.append( [ "Linear SVC", LinearSVC() ] )
    classifiers.append( [ "RBF Kernel SVC", SVC(kernel='rbf', C=3, gamma=.01) ] )
    #classifiers.append( [ "Sigmoid Kernel SVC", SVC(kernel='sigmoid', C=10, gamma=.005) ] )
    #classifiers.append( [ "Linear Kernel SVC", SVC(kernel='linear', C=10, gamma=.2) ] )
    classifiers.append( [ "NU SVC", NuSVC() ] )
    
    classifierSelected = ensemble(classifiers, X_train, y_train, X_test, y_test)
    
    log("Completed training / cross-validation...")

    log("Starting test data analysis...")

    testFileRaw = "data/test.csv"    
    testFileNpy = "data/test.npy"    
    X = load(testFileRaw, testFileNpy)
    X = X/255.0*2 - 1
    y_pred = predict(classifierSelected, X)
    print(y_pred.shape)
    print(y_pred)
    
    predictionsCSV = "data/predictions.csv"
    np.savetxt(predictionsCSV, y_pred, fmt="%u", delimiter=",")
    
    log("Completed prediction, output: {0}".format(predictionsCSV))

    log("Completed test data analysis...")

    log("Completed mainline")
    
if __name__ == "__main__":
    main()    

