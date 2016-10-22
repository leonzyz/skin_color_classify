#!/usr/bin/python
import sys
path2cv='/usr/local/Cellar/opencv3/HEAD/lib/python2.7/site-packages'
sys.path.append(path2cv)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
import cv2
from sknn.mlp import Regressor
from sknn.mlp import Layer
import skin_color_based_segment as sg

flagConvert=True
data,labels=sg.ReadTrainData(1.0,flagConvert)
print data.shape
print "Tree ===="
clf_tree=sg.Training(data,labels,'RGB','Tree')
print "NN ===="

data=sg.ConvertColor(data,'RGB')
trainData,testData,trainlabels,testlabels=train_test_split(data,labels,test_size=0.2)

traindata_s=trainData[0:8000]
trainlabels_s=trainlabels[0:8000]
testdata_s=testData[0:4000]
testlabels_s=testlabels[0:4000]
nn=Regressor(layers=[Layer("Linear",units=6),Layer("Linear")],learning_rate=0.02,n_iter=3)
nn.fit(traindata_s,trainlabels_s)
print nn.score(testData_s,testlabels_s)
