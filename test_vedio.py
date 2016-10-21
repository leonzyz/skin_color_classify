#!/usr/bin/python
import pickle
import numpy as np
from sklearn import linear_model
import color_space_trans as ct


pkl_file=open('logistical_classifier.pkl','rb')
#pkl_file=open('logistical_classifier_Lab.pkl','rb')
logreg=pickle.load(pkl_file)
pkl_file.close()

import cv2
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
org_size=frame.shape
print frame.shape
print frame[0:2,0:2,:]


def get_segment_frame(frame_in,classifier):
	maskframe=frame_in.copy()
	total_len=frame_in.shape[0]
	#labframe=cv2.cvtColor(frame_in,cv2.COLOR_RGB2Lab)
	for idx in range(total_len):
		org_line=frame_in[idx,:,:]
		line=ct.RGB_norm(org_line.copy())
		#line_lab=labframe[idx,:,:]
		#line_lab_norm=ct.Lab_norm(line_lab)
		#line_test=np.column_stack((line,line_lab_norm[:,1:3]))
		#testout=classifier.predict(line_test)
		testout=classifier.predict(line)
		testout=(testout+1)/2*255
		mask=np.array([testout,testout,testout])
		maskframe[idx,:,:]=np.transpose(mask)

	return maskframe



cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('seg out',cv2.WINDOW_NORMAL)
while True:
	ret,frame=cap.read()
	outframe=cv2.resize(frame,(org_size[1]/2,org_size[0]/2))
	#outframe=frame
	maskframe=get_segment_frame(outframe,logreg)

	cv2.imshow('seg out',maskframe)
	cv2.imshow('frame',outframe)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break



cap.release()
cv2.destroyAllWindows()
