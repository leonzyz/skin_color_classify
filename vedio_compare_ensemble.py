#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import cv2
import skin_color_based_segment as sg

cap=cv2.VideoCapture(0)

#flagColorSpace='BGR'
#flagColorSpace='Lab'
#flagColorSpace='BGRab'
#flagColorSpace='HSV'
flagConvert=True
#flagCarmera=True
#flagAlg='Logistical'
flagAlg='Tree'
data,labels=sg.ReadTrainData(1.0,flagConvert)

#ref
eclf_BGRab=sg.Training(data,labels,'Lab','Voting')

#new
#clf_BGR_tree=sg.Training(data,labels,'BGR','Tree')
#clf_BGR_log=sg.Training(data,labels,'BGR','Logistical')

estimators=[('Labtree',sg.colortransform('Lab')),('clf',tree.DecisionTreeClassifier(criterion='entropy'))]
pipe_LAB_tree=Pipeline(estimators)
estimators=[('HSVtree',sg.colortransform('HSV')),('clf',tree.DecisionTreeClassifier(criterion='entropy'))]
pipe_HSV_tree=Pipeline(estimators)
estimators=[('Lablog',sg.colortransform('Lab')),('clf',linear_model.LogisticRegression())]
pipe_LAB_log=Pipeline(estimators)
clf_vote=VotingClassifier(estimators=[('BGRtree',tree.DecisionTreeClassifier(criterion='entropy')),('Labtree',pipe_LAB_tree),('HSVtree',pipe_HSV_tree),('Lablog',pipe_LAB_log)],voting='soft',weights=[1,1,1,3])

clf_vote.fit(data,labels)


'''
clf_BGRab_tree=sg.Training(data,labels,'BGRab','Tree')
clf_BGRab_log=sg.Training(data,labels,'BGRab','Logistical')
'''



cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('BGRab_vote',cv2.WINDOW_NORMAL)
#cv2.namedWindow('pipe',cv2.WINDOW_NORMAL)
cv2.namedWindow('ensemble',cv2.WINDOW_NORMAL)
'''
cv2.namedWindow('BGR_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('Lab_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('BGRab_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('BGR_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('Lab_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('BGRab_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('combine',cv2.WINDOW_NORMAL)
'''
while True:
	ret,frame=cap.read()
	if True:
		frame=cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))

	#grap_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	BGRab_vote_frame=sg.ApplyFrame(eclf_BGRab,frame,'Lab',flagConvert)
	#pipe_frame=sg.ApplyFrame(pipe,frame,'BGR',flagConvert)
	ensemble_frame=sg.ApplyFrame_BGR(clf_vote,frame,flagConvert)
	'''
	BGR_tree_frame=sg.ApplyFrame(clf_BGR_tree,frame,'BGR',flagConvert)
	Lab_tree_frame=sg.ApplyFrame(clf_LAB_tree,frame,'Lab',flagConvert)
	HSV_tree_frame=sg.ApplyFrame(clf_HSV_tree,frame,'HSV',flagConvert)
	BGRab_tree_frame=sg.ApplyFrame(clf_BGRab_tree,frame,'BGRab',flagConvert)
	BGR_log_frame=sg.ApplyFrame(clf_BGR_log,frame,'BGR',flagConvert)
	Lab_log_frame=sg.ApplyFrame(clf_LAB_log,frame,'Lab',flagConvert)
	HSV_log_frame=sg.ApplyFrame(clf_HSV_log,frame,'HSV',flagConvert)
	BGRab_log_frame=sg.ApplyFrame(clf_BGRab_log,frame,'BGRab',flagConvert)
	combine_frame=np.uint8((np.int32(BGR_tree_frame)+np.int32(Lab_tree_frame)+np.int32(HSV_tree_frame)+np.int32(BGRab_tree_frame)+np.int32(BGR_log_frame)+np.int32(Lab_log_frame)+np.int32(HSV_log_frame)+np.int32(BGRab_log_frame))/8)
	'''

	#cv2.imshow('frame',frame)
	#cv2.imshow('Lab_tree',Lab_tree_frame)
	cv2.imshow('BGRab_vote',BGRab_vote_frame)
	#cv2.imshow('pipe',pipe_frame)
	cv2.imshow('ensemble',ensemble_frame)
	'''
	cv2.imshow('BGR_tree',BGR_tree_frame)
	cv2.imshow('HSV_tree',HSV_tree_frame)
	cv2.imshow('BGRab_tree',BGRab_tree_frame)
	cv2.imshow('BGRab_log',BGRab_log_frame)
	cv2.imshow('BGR_log',BGR_log_frame)
	cv2.imshow('HSV_log',HSV_log_frame)
	cv2.imshow('Lab_log',Lab_log_frame)
	cv2.imshow('combine',combine_frame)
	'''
	#while True:
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

