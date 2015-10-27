import numpy as np
import time
import sys
import os.path

print("Starting...")

trainingFileRaw = "data/train.csv"
trainingFileNpy = "data/train.npy"

# Load the data from file and, if necessary, save RAW file as NPY file to speed future use
if( os.path.exists(trainingFileNpy) ):
    print("{0}: Loading NPY dataset: {1}".format(time.asctime(), trainingFileNpy) )
    datasetFull = np.load(trainingFileNpy)
else:
    print("Loading RAW file")
    print("{0}: Loading RAW dataset: {1}".format(time.asctime(), trainingFileRaw) )
    datasetFull = np.loadtxt(open(trainingFileRaw,"rb"),delimiter=",",skiprows=1)
    print("{0}: Saving RAW data into NPY dataset: {1}".format(time.asctime(), trainingFileNpy) )
    np.save(trainingFileNpy, datasetFull)
    
# Take a small subset of testing hypothesis to lower run-time
numRows, numFeatures = datasetFull.shape
datasetPct = 0.05
rowsUsed = int(numRows * datasetPct)
dataset = datasetFull[0:rowsUsed,:]

# Get metrics on the dataset
m, n = dataset.shape

y = dataset[:,0]
X = dataset[:,1:m]

print("{0}: Splitting dataset into training and test data".format(time.asctime()) )
from sklearn.cross_validation import train_test_split
pctTest = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pctTest, random_state=42)

mTrain, nTrain = X_train.shape
mTest, nTest = X_test.shape

print("\nKey Metrics")
print("Dataset (full) size: \t\t{0}".format(numRows))
print("Num features: \t\t\t{0}".format(numFeatures))
print("PCT of full data set used: \t{0}".format(datasetPct))
print("Num rows: \t\t\t{0}".format(m))
print("Training PCT: \t\t\t{0} \tNum training: \t{1}".format(1-pctTest, mTrain))
print("Test PCT: \t\t\t{0} \tNum test: \t{1}".format(pctTest, mTest))

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


confusionMatrix = np.zeros((10, 10))
cscore = np.zeros((8, 10))

print("\n{0}: Executing random forest classifier".format(time.asctime()) )
maxDepth=100
numEstimators=100
maxFeatures=100
rfc = RandomForestClassifier(max_depth=maxDepth, n_estimators=numEstimators, max_features=maxFeatures)
rfc.fit(X_train, y_train)
#print("RFC max depth: {0}, num estimators: {1}, max features: {2} ".format(maxDepth, numEstimators, maxFeatures))
print("RFC train: ", rfc.score(X_train, y_train))
print("RFC test:  ", rfc.score(X_test, y_test))

datasetTest = np.loadtxt(open("data/test.csv","rb"),delimiter=",",skiprows=1)
y_pred = rfc.predict(datasetTest)
print(y_pred[0:100])
sys.exit()


y_pred = rfc.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
crpt = classification_report(y_test, y_pred)
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
print("f1 score: ", f1_score(y_test, y_pred, average=None))

cscore[0] = f1_score(y_test, y_pred, average=None)
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing nearest neighbour classifier".format(time.asctime()) )
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
print("KNN train: ", knn.score(X_train, y_train))
print("KNN test:  ", knn.score(X_test, y_test))

y_pred = knn.predict(X_test)
y_pred = np.around(y_pred, decimals=0)
#print("y_pred: ", y_pred)
#print("y_test: ", y_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))

cscore[1] = f1_score(y_test, y_pred, average=None)
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing decision tree classifier".format(time.asctime()) )
maxDepth=10
dtc = DecisionTreeClassifier(max_depth=maxDepth)
dtc.fit(X_train, y_train)
print("DTC max depth: {0}".format(maxDepth))
print("DTC train: ", dtc.score(X_train, y_train))
print("DTC test:  ", dtc.score(X_test, y_test))

y_pred = dtc.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing adaboost classifier".format(time.asctime()) )
abc = AdaBoostClassifier()
abc.fit(X_train, y_train)
print("AdaBoost train: ", abc.score(X_train, y_train))
print("AdaBoost test:  ", abc.score(X_test, y_test))

y_pred = abc.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing lda classifier".format(time.asctime()) )
lda = LDA()
lda.fit(X_train, y_train)
print("LDA train: ", lda.score(X_train, y_train))
print("LDA test:  ", lda.score(X_test, y_test))

y_pred = lda.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing Gaussian naive-bayes classifier".format(time.asctime()) )
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("GNB train: ", gnb.score(X_train, y_train))
print("GNB test:  ", gnb.score(X_test, y_test))

y_pred = gnb.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing qda classifier".format(time.asctime()) )
qda = QDA()
qda.fit(X_train, y_train)
print("QDA train: ", qda.score(X_train, y_train))
print("QDA test:  ", qda.score(X_test, y_test))

y_pred = qda.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing logistic regression classifier".format(time.asctime()) )
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("LR train: ", lr.score(X_train, y_train))
print("LR test:  ", lr.score(X_test, y_test))

y_pred = lr.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)

print("\n{0}: Executing linear SVC classifier".format(time.asctime()) )
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
print("LSVC train: ", lsvc.score(X_train, y_train))
print("LSVC test:  ", lsvc.score(X_test, y_test))

print("\n{0}: Executing svc classifier".format(time.asctime()) )
#C=1.0
#gamma=1/numFeatures
#svc = SVC(C=C, gamma=gamma)
#C=1.0
#gamma=1/numFeatures
#svc = SVC(C=C, gamma=gamma)
lsvc.fit(X_train, y_train)
#print("SVC C: \t{0}, \tgamma: \t{1} ".format(C, gamma))
print("SVC train: ", lsvc.score(X_train, y_train))
print("SVC test:  ", lsvc.score(X_test, y_test))

y_pred = lsvc.predict(X_test)
print("\nScoring Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix (horz: predicted / vert: actual")
print(confusion_matrix(y_test, y_pred))
confusionMatrix = confusionMatrix + confusion_matrix(y_test, y_pred)


