# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:29:28 2015

@author: Davis
"""

import numpy as np
import logging
import sys
import os.path

from Loader import Loader

class DigitLoader(Loader):
    
    def __init__(self, pathNpy, pathRaw):
        self.paths = [pathNpy, pathRaw]
        super(DigitLoader, self).__init__(self.paths)
        logging.getLogger('').handlers = []
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        print(self.getPaths())
        
    def getPaths(self):
        return self.paths
        
    def loadDigits(self, paths=[]):
        pathNpy = self.paths[0]
        pathRaw = self.paths[1]
        if( os.path.exists(pathNpy) ):
            logging.info("Loading NPY dataset: {0}".format(pathNpy) )
            datasetFull = np.load(pathNpy)
        else:
            logging.info("Loading RAW file")
            logging.info("Loading RAW dataset: {0}".format(pathRaw) )
            datasetFull = np.loadtxt(open(pathRaw,"rb"),delimiter=",",skiprows=1)
            logging.info("Saving RAW data into NPY dataset: {0}".format(pathNpy) )
            np.save(pathNpy, datasetFull)
        return datasetFull
        
    def load(self):
        return super(DigitLoader, self).load(self.loadDigits)
        
    
        