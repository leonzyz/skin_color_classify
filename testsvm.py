#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
import cv2
import skin_color_based_segment as sg
from sklearn import svm

flagConvert=True
data,labels=sg.ReadTrainData(1.0,flagConvert)
print data.shape
print "Tree ===="
clf_tree=sg.Training(data,labels,'RGB','Tree')
print "NN ===="

data=sg.ConvertColor(data,'RGB')
trainData,testData,trainlabels,testlabels=train_test_split(data,labels,test_size=0.2)

clf_svm=svm.SVC()
clf_svm.fit(trainData,trainlabels)
print clf_svm.score(testData,testlabels)

outfile=open('clf_svm_full.pkl','w')
pickle.dump(clf_svm,outfile)
outfile.close()
