# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:02:03 2015

@author: Davis
"""

from DigitVisualizer import DigitVisualizer

#testing launch script

start = 1
end = 2

trainingFileRaw = "data/train.csv"
trainingFileNpy = "data/train.npy"   

visualizer = DigitVisualizer(0, 1, trainingFileNpy,trainingFileRaw)
visualizer.run()