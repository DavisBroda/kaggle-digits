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

from PIL import Image
    
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

def log(message):
    logging.info(message)
    
def execute(dataset, items):
    log("Executing...")
    
    log("items: {0}".format(items))

    m, n = dataset.shape
    targets = dataset[:,0].astype(int)
    images = dataset[:,1:n].astype(int)
    
    for i in items:
        image = images[i]
        target = targets[i]
        log("Image: {0}".format(i))
        visualize(target, image)
        view(image)
     
def visualize(target, flatImage):
    log("Visualizing...")
    
    log("Target: {0}".format(target))
    
    image = flatImage.astype(int).reshape(28,28).astype(int)
    m,n = image.shape

    str = "\n\n     "
    for i in range(n):
        str = str + "{0:3d} ".format(i)

    for i in range(m):
        row = image[i]
        
        rowstr = ""
        for j in range(n):
            value = image[i][j]
            rowstr = rowstr + "{0:03d}.".format(value)

        str = str + "\n{0:3d}: {1}".format(i, rowstr)

    str = str + "\n"
    
    log(str)

def view(flatImage):
    
    m,n = 28,28
    img = Image.new( 'RGB', (m,n), "black") # create a new black image
    pixels = img.load() # create the pixel map
    
    #for i in range(img.size[0]):    # for every pixel:
    #    for j in range(img.size[1]):
    #        pixels[i,j] = (i, j, 100) # set the colour accordingly 
    #img.show()
        
    image = flatImage.astype(int).reshape(m,n).astype(int)
    m,n = image.shape

    for i in range(m):
        for j in range(n):
            value = image[i][j]

            # Equal RGB values (255 == white and 0 == black)
            pixels[i,j] = [value, value, value]

    img.show()
     
def main():
    
    print( 'Number of arguments: {0}'.format(len(sys.argv)) )
    print( 'Argument List: {0}'.format(str(sys.argv)) )
    
    start = 1
    if len(sys.argv) > 1:
        start = sys.argv[1]
    
    end = start + 1
    if len(sys.argv) > 2:
        end = sys.argv[2]
    
    logging.getLogger('').handlers = []
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log("Started mainline")
    
    trainingFileRaw = "data/train.csv"
    trainingFileNpy = "data/train.npy"   
    dataset = load(trainingFileRaw, trainingFileNpy)
    m, n = dataset.shape

    log("Full data set: rows: {0}, features: {1}".format(m,n))    

    predictions = execute(dataset, range(start, end))
    
    log("Completed mainline")
    
if __name__ == "__main__":
    main()    

