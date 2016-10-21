#!/usr/bin/python
# RGB+Lab color based segmentation, logistical regression
import pickle
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn import linear_model
import color_space_trans as ct
import cv2

pkl_file=open('inputset.pkl','rb')
input_set=pickle.load(pkl_file)
pkl_file.close()
total_dataset_size=len(input_set)

data_set=np.array(input_set)

cross_validation_ratio=0.2
valid_num=int(total_dataset_size*cross_validation_ratio)

tran_set_idx=range(total_dataset_size)
valid_set_idx=[]
tran_set_len=total_dataset_size
for idx in range(valid_num):
	idx=rd.randint(0,tran_set_len-1)
	valid_set_idx.append(tran_set_idx[idx])
	del tran_set_idx[idx]
	tran_set_len-=1

rd.shuffle(tran_set_idx)

valid_set=data_set[valid_set_idx]
tran_set=data_set[tran_set_idx]


classifier=linear_model.LogisticRegression()
sample=tran_set[:,0:3]
sample_lab_tmp=cv2.cvtColor(np.array([sample],dtype=np.uint8),cv2.COLOR_RGB2Lab)
sample_lab=sample_lab_tmp[0]
print sample[0:10,:]
print sample_lab[0:10,:]
sample=ct.RGB_norm(sample)
sample_lab_norm=ct.Lab_norm(sample_lab)
sample_rgb_lab=np.column_stack((sample,sample_lab_norm[:,1:3]))
print sample_rgb_lab[0:10,:]

target=tran_set[:,3]
#target=(target-1.5)*-2.0
target=target-1.0
classifier.fit(sample_rgb_lab,target)

test=valid_set[:,0:3]
print test[0:10,:]
test_lab_tmp=cv2.cvtColor(np.array([test],dtype=np.uint8),cv2.COLOR_RGB2Lab)
test_lab=test_lab_tmp[0]
print test_lab[0:10,:]
test=ct.RGB_norm(test)
test_lab_norm=ct.Lab_norm(test_lab)
test_rgb_lab=np.column_stack((test,test_lab_norm[:,1:3]))
print test_rgb_lab[0:10,:]
#valid_out=(valid_set[:,3]-1.5)*-2.0
valid_out=valid_set[:,3]-1.0
x=classifier.predict(test_rgb_lab)

corret_set=x==valid_out
correct_ratio=(corret_set==True).sum()/float(valid_num)
'''
tp=np.logical_and(valid_out==1,x==1).sum()
fp=np.logical_and(valid_out==-1,x==1).sum()
tn=np.logical_and(valid_out==-1,x==-1).sum()
fn=np.logical_and(valid_out==-1,x==1).sum()
'''

tp=np.logical_and(valid_out==0,x==0).sum()
fp=np.logical_and(valid_out==1,x==0).sum()
tn=np.logical_and(valid_out==1,x==1).sum()
fn=np.logical_and(valid_out==0,x==1).sum()
'''
tpr
fpr
tnr
fnr
'''
fpr=fp/float(fp+tn)
tpr=tp/float(tp+fn)
print "%d %d %d %d" %(tp,fp,tn,fn)
print "FPR:%f,TPR:%f" %(fpr*100.0,tpr*100.0)
print "correct rate:%f" % (correct_ratio*100.0)

outfile=open('logistical_classifier_lab.pkl','wb')
#cPickle.dump(classifier,outfile)
pickle.dump(classifier,outfile)
outfile.close()
