#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
import cv2
import skin_color_based_segment as sg

cap=cv2.VideoCapture(0)

#flagColorSpace='RGB'
#flagColorSpace='Lab'
#flagColorSpace='RGBab'
#flagColorSpace='HSV'
flagConvert=True
#flagCarmera=True
#flagAlg='Logistical'
flagAlg='Tree'
data,labels=sg.ReadTrainData(1.0,flagConvert)
eclf_RGBab=sg.Training(data,labels,'Lab','Voting')
'''
clf_RGB_tree=sg.Training(data,labels,'RGB','Tree')
clf_LAB_tree=sg.Training(data,labels,'Lab','Tree')
clf_HSV_tree=sg.Training(data,labels,'HSV','Tree')
clf_RGBab_tree=sg.Training(data,labels,'RGBab','Tree')
clf_RGB_log=sg.Training(data,labels,'RGB','Logistical')
clf_HSV_log=sg.Training(data,labels,'HSV','Logistical')
clf_LAB_log=sg.Training(data,labels,'Lab','Logistical')
clf_RGBab_log=sg.Training(data,labels,'RGBab','Logistical')
'''



cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('RGBab_vote',cv2.WINDOW_NORMAL)
'''
cv2.namedWindow('RGB_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('Lab_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('RGBab_tree',cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('RGB_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('Lab_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('RGBab_log',cv2.WINDOW_NORMAL)
cv2.namedWindow('combine',cv2.WINDOW_NORMAL)
'''
while True:
	ret,frame=cap.read()
	if True:
		frame=cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))

	grap_frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	RGBab_vote_frame=sg.ApplyFrame(eclf_RGBab,frame,'Lab',flagConvert)
	'''
	RGB_tree_frame=sg.ApplyFrame(clf_RGB_tree,frame,'RGB',flagConvert)
	Lab_tree_frame=sg.ApplyFrame(clf_LAB_tree,frame,'Lab',flagConvert)
	HSV_tree_frame=sg.ApplyFrame(clf_HSV_tree,frame,'HSV',flagConvert)
	RGBab_tree_frame=sg.ApplyFrame(clf_RGBab_tree,frame,'RGBab',flagConvert)
	RGB_log_frame=sg.ApplyFrame(clf_RGB_log,frame,'RGB',flagConvert)
	Lab_log_frame=sg.ApplyFrame(clf_LAB_log,frame,'Lab',flagConvert)
	HSV_log_frame=sg.ApplyFrame(clf_HSV_log,frame,'HSV',flagConvert)
	RGBab_log_frame=sg.ApplyFrame(clf_RGBab_log,frame,'RGBab',flagConvert)
	combine_frame=np.uint8((np.int32(RGB_tree_frame)+np.int32(Lab_tree_frame)+np.int32(HSV_tree_frame)+np.int32(RGBab_tree_frame)+np.int32(RGB_log_frame)+np.int32(Lab_log_frame)+np.int32(HSV_log_frame)+np.int32(RGBab_log_frame))/8)
	'''

	#cv2.imshow('frame',frame)
	cv2.imshow('Lab_tree',Lab_tree_frame)
	cv2.imshow('RGBab_vote',RGBab_vote_frame)
	'''
	cv2.imshow('RGB_tree',RGB_tree_frame)
	cv2.imshow('HSV_tree',HSV_tree_frame)
	cv2.imshow('RGBab_tree',RGBab_tree_frame)
	cv2.imshow('RGBab_log',RGBab_log_frame)
	cv2.imshow('RGB_log',RGB_log_frame)
	cv2.imshow('HSV_log',HSV_log_frame)
	cv2.imshow('Lab_log',Lab_log_frame)
	cv2.imshow('combine',combine_frame)
	'''
	#while True:
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

