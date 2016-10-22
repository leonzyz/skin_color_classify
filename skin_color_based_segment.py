#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
import cv2
from sklearn import svm


def ReadTrainData(ratio,flagConvert):
	src=np.genfromtxt('Skin_NonSkin.txt',dtype=np.uint8)
	total_len=src.shape[0]
	outlen=int(total_len*ratio)
	data=src[:outlen,0:3]
	labels=src[:outlen,3]
	#conver True 1 False 2 to True 1 False 0
	if flagConvert:
		labels=2-labels
	return data,labels
	

def ConvertColor(rgb,flag):
	dem=rgb.shape
	if len(dem)==2:
		tmp=np.reshape(rgb,(dem[0],1,3))
	else:
		tmp=rgb

	if flag=='HSV':
		tmp=cv2.cvtColor(tmp,cv2.COLOR_RGB2HSV)
	elif flag=='Lab':
		tmp=cv2.cvtColor(tmp,cv2.COLOR_RGB2Lab)
	elif flag=='RGBab':
		tmplab=cv2.cvtColor(tmp,cv2.COLOR_RGB2Lab)
		if len(dem)==2:
			tmplab=np.reshape(tmplab,dem)
			tmp=np.column_stack((rgb,tmplab[:,1:3]))
			tmp=rgb
		else:
			tmplab=np.reshape(tmplab,(dem[0]*dem[1],3))
			tmp=np.column_stack((np.reshape(rgb,(dem[0]*dem[1],3)),tmplab[:,1:3]))
			tmp=np.reshape(tmp,(dem[0],dem[1],dem[2]+2))
			tmp=rgb
	else:
		return rgb

	return np.reshape(tmp,dem)

def Training(data,labels,flagColorSpace,flagAlg):
	data=ConvertColor(data,flagColorSpace)

	trainData,testData,trainlabels,testlabels=train_test_split(data,labels,test_size=0.2)

	if flagAlg=='Tree':
		clf=tree.DecisionTreeClassifier(criterion='entropy')
	elif flagAlg=='Logistical':
		clf=linear_model.LogisticRegression()
	elif flagAlg=='Linear':
		clf=linear_model.LinearRegression()
	elif flagAlg=='Voting':
		#clf=VotingClassifier(estimators=[('tree',tree.DecisionTreeClassifier(criterion='entropy')),('logreg',linear_model.LogisticRegression())],voting='soft',weights=[2,1])
		clf=VotingClassifier(estimators=[('tree',tree.DecisionTreeClassifier(criterion='entropy')),('logreg',linear_model.LogisticRegression())],voting='soft',weights=[3,1])
	else:
		clf=linear_model.LogisticRegression()

	clf.fit(trainData,trainlabels)
	if flagAlg=='Tree':
		print clf.feature_importances_
	print clf.score(testData,testlabels)

	return clf

def ApplyFrame(clf,framein,flagColorSpace,flagConvert):
	frameout=framein.copy()
	framesize=frameout.shape
	ConvetedFrame=ConvertColor(frameout,flagColorSpace)
	frame_1d=np.reshape(ConvetedFrame,(framesize[0]*framesize[1],framesize[2]))
	testout=clf.predict(frame_1d)
	if flagConvert:
		testout=testout*255
	else:
		testout=(2-testout)*255
	mask=np.array([testout,testout,testout])
	frameout=np.reshape(np.transpose(mask),framesize)
	return frameout

def TestVedioe(clf,flagColorSpace,flagCarmera,flagConvert,path):
	if flagCarmera:
		cap=cv2.VideoCapture(0)
	else:
		cap=cv2.VideoCapture(path)
	

	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.namedWindow('seg out',cv2.WINDOW_NORMAL)
	while True:
		ret,frame=cap.read()
		if True:
			frame=cv2.resize(frame,(frame.shape[1]/2,frame.shape[0]/2))
		outframe=ApplyFrame(clf,frame,flagColorSpace,flagConvert)

		cv2.imshow('seg out',outframe)
		cv2.imshow('frame',frame)
		#while True:
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break

#flagColorSpace='RGB'
#flagColorSpace='Lab'
flagColorSpace='RGBab'
#flagColorSpace='HSV'
flagConvert=True
flagCarmera=True
#flagAlg='Logistical'
#flagAlg='Tree'
data,labels=ReadTrainData(1.0,flagConvert)
#clf=Training(data,labels,flagColorSpace,flagAlg)
#TestVedioe(clf,flagColorSpace,flagCarmera,flagConvert,'')
#clf=Training(data,labels,flagColorSpace,'Linear')
eclf=Training(data,labels,flagColorSpace,'Voting')



