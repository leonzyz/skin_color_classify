#!/usr/bin/python
import pickle
import numpy as np
from sklearn import linear_model
import color_space_trans as ct


#pkl_file=open('logistical_classifier.pkl','rb')
pkl_file=open('logistical_classifier_Lab.pkl','rb')
logreg=pickle.load(pkl_file)
pkl_file.close()

import cv2
frame=cv2.imread('face.png')
#cap=cv2.VideoCapture('./test.mp4')


def get_segment_frame(frame_in,classifier):
	maskframe=frame_in.copy()
	mf_row=maskframe.shape[1]
	mf_col=maskframe.shape[0]
	mf_pixel=mf_row*mf_col
	maskframe_1d=np.reshape(maskframe,(mf_pixel,3))
	maskframe_lab=cv2.cvtColor(maskframe,cv2.COLOR_RGB2Lab)
	maskframe_1d_lab=np.reshape(maskframe_lab,(mf_pixel,3))
	maskframe_1d_rgb_normal=ct.RGB_norm(maskframe_1d)
	maskframe_1d_lab_normal=ct.Lab_norm(maskframe_1d_lab)
	maskframe_vector=np.column_stack((maskframe_1d_rgb_normal,maskframe_1d_lab_normal[:,1:3]))
	testout=classifier.predict(maskframe_vector)
	testout=(1-testout)*255
	mask=np.array([testout,testout,testout])
	maskframe=np.reshape(np.transpose(mask),(mf_col,mf_row,3))
	return maskframe



cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('seg out',cv2.WINDOW_NORMAL)
while True:
	maskframe=get_segment_frame(frame,logreg)

	cv2.imshow('seg out',maskframe)
	cv2.imshow('frame',frame)

	#while True:
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break



cap.release()
cv2.destroyAllWindows()
