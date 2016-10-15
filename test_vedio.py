#!/usr/bin/python
import pickle
import numpy as np
from sklearn import linear_model


pkl_file=open('logistical_classifier.pkl','rb')
logreg=pickle.load(pkl_file)
pkl_file.close()

import cv2
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
org_size=frame.shape


def get_segment_frame(frame_in,classifier):
	maskframe=frame_in.copy()
	total_len=frame_in.shape[0]
	for idx in range(total_len):
		org_line=frame_in[idx,:,:]
		line=org_line.copy()/256.0
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
	maskframe=get_segment_frame(outframe,logreg)

	cv2.imshow('seg out',maskframe)
	cv2.imshow('frame',outframe)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break



cap.release()
cv2.destroyAllWindows()
