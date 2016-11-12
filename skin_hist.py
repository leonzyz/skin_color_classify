#!/usr/bin/python

#print "hello"

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import skin_color_based_segment as sg


zv=np.ones(shape=(256,256),dtype=np.uint8)*128
z_plane=np.zeros(shape=(256,256,3),dtype=np.uint8)
x_range=np.array([range(0,256)],dtype=np.uint8)
y_range=np.array([range(0,256)],dtype=np.uint8)
xv,yv=np.meshgrid(x_range,y_range)
z_plane[:,:,0]=zv
z_plane[:,:,1]=xv
z_plane[:,:,2]=yv
RGB_plane=cv2.cvtColor(z_plane,cv2.COLOR_Lab2RGB)
"""
fig=plt.figure()
ax=fig.add_subplot(111)
ax.imshow(RGB_plane)
plt.show()
"""

flagConvert=True
data,label=sg.ReadTrainData(1.0,flagConvert)
lab_data=sg.BGR2Lab(data)
eclf_Lab=sg.Training(data,label,'Lab','Voting')
clf_Lab_tree=sg.Training(data,label,'Lab','Tree')

total_len=lab_data.shape[0]
#print total_len

#buff=np.zeros(shape=(256,256),dtype=np.float32)
buff=np.ones(shape=(256,256),dtype=np.float32)
buff_count=np.ones(shape=(256,256),dtype=np.float32)
for idx in range(0,total_len):
	x=lab_data[idx,1]
	y=lab_data[idx,2]
	buff[x,y]+=label[idx]
	buff_count[x,y]+=1

#max_val=np.amax(buff)
#scaling=1.0/max_val
buff_log=np.log(buff)
buff_count_log=np.log(buff_count)

#fig=plt.figure()
#ax=fig.add_subplot(111)

fig2=plt.figure()
cx=fig2.add_subplot(111)
"""
ax.plot_surface(xv[100:150,70:120],yv[100:150,70:120],buff_log[100:150,70:120],cmap=mpl.cm.coolwarm,rstride=1,cstride=1)
bx.plot_surface(xv,yv,buff_log,cmap=mpl.cm.coolwarm,rstride=5,cstride=5)
"""

cx.imshow(RGB_plane,origin='lower')
cx.contour(xv,yv,buff_log,cmap=mpl.cm.coolwarm,rstride=5,cstride=5)
imout=sg.ApplyFrame(eclf_Lab,cv2.cvtColor(RGB_plane.copy(),cv2.COLOR_RGB2BGR),'Lab',flagConvert)
fig3=plt.figure()
ax=fig3.add_subplot(111)
#ax.imshow(imout,origin='lower')
#ret,thresh=cv2.threshold(np.array(imout,dtype=np.uint8),127,255,0)
ret,thresh=cv2.threshold(cv2.cvtColor(imout,cv2.COLOR_RGB2GRAY),127,255,0)
image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
RGB_plane_contour=cv2.drawContours(RGB_plane.copy(),contours,-1,(0,0,0),3)
ax.imshow(RGB_plane_contour,origin='lower')
#ax.contour(xv,yv,buff_log,cmap=mpl.cm.coolwarm,rstride=5,cstride=5)
ax.contour(yv,xv,buff_log,cmap=mpl.cm.coolwarm,rstride=5,cstride=5)

imout=sg.ApplyFrame(clf_Lab_tree,cv2.cvtColor(RGB_plane.copy(),cv2.COLOR_RGB2BGR),'Lab',flagConvert)
fig4=plt.figure()
ax=fig4.add_subplot(111)
ret,thresh=cv2.threshold(cv2.cvtColor(imout,cv2.COLOR_RGB2GRAY),127,255,0)
image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
RGB_plane_contour=cv2.drawContours(RGB_plane.copy(),contours,-1,(255,255,255),3)
ax.imshow(RGB_plane_contour,origin='lower')

fig5=plt.figure()
cx=fig5.add_subplot(111)
cx.imshow(RGB_plane,origin='lower')
cx.contour(yv,xv,buff_count_log,cmap=mpl.cm.coolwarm,rstride=5,cstride=5)

plt.show()
