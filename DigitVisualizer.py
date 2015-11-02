# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:54:24 2015

@author: Davis Broda
"""

import numpy as np
import time
import logging
import os.path
import sys
from PIL import Image

from DigitLoader import DigitLoader

#Attempt to put visualize into a proper class

class DigitVisualizer:
    
    def __init__(self, start, end, dataRawPath, dataNpyPath):
        self.start = start
        self.end = end
        self.dataRawPath = dataRawPath
        self.dataNpyPath = dataNpyPath
        
        #set up logging
        logging.getLogger('').handlers = []
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    def log(self, message):
        logging.info(message)
    
    def visualizeAll(self, dataset, items):
        self.log("Executing...")
        
        self.log("items: {0}".format(items))
    
        m, n = dataset.shape
        targets = dataset[:,0].astype(int)
        images = dataset[:,1:n].astype(int)
        
        for i in items:
            image = images[i]
            target = targets[i]
            self.log("Image: {0}".format(i))
            self.visualize(target, image)
            self.view(image)
         
    def visualize(self, target, flatImage):
        self.log("Visualizing...")
        
        self.log("Target: {0}".format(target))
        
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
        
        self.log(str)
    
    def view(self,flatImage):
        
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
                pixels[i,j] = (value, value, value)
    
        img.show()
         
    def run(self):
        
        #print( 'Number of arguments: {0}'.format(len(sys.argv)) )
        #print( 'Argument List: {0}'.format(str(sys.argv)) )
        
#        start = 1
#        if len(sys.argv) > 1:
#            start = int(sys.argv[1])
#        
#        end = start + 1
#        if len(sys.argv) > 2:
#            end = int(sys.argv[2])
        
        
        self.log("Started mainline")
        
#        trainingFileRaw = "data/train.csv"
#        trainingFileNpy = "data/train.npy"   
        loader = DigitLoader(self.dataRawPath, self.dataNpyPath)        
        
        dataset = loader.load()
        print(dataset[:,1])
        m, n = dataset.shape
    
        self.log("Full data set: rows: {0}, features: {1}".format(m,n))    
    
        predictions = self.visualizeAll(dataset, range(self.start, self.end))
        
        self.log("Completed mainline")