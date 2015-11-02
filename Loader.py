# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:19:47 2015

@author: Davis
"""

import logging
import sys

class Loader:
    
    def __init__(self, paths=[]):
        self.filePaths = paths
        logging.getLogger('').handlers = []
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def load(self, loadFunction, *args):
        return loadFunction(self.filePaths, *args)
        