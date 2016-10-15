#!/usr/bin/python
import pickle
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from sklearn import linear_model


pkl_file=open('inputset.pkl','rb')
input_set=pickle.load(pkl_file)
pkl_file.close()
total_dataset_size=len(input_set)
data_set=np.array(input_set)
sample=data_set[:,0:3]
sample=sample/256.0
valid_out=data_set[:,3]
valid_out=(valid_out-1.5)*-2.0
pkl_file=open('logistical_classifier.pkl','rb')
classifier=pickle.load(pkl_file)
pkl_file.close()

x=classifier.predict(sample)

correct_ratio=(x==valid_out).sum()/float(total_dataset_size)
tp=np.logical_and(valid_out==1,x==1).sum()
fp=np.logical_and(valid_out==-1,x==1).sum()
tn=np.logical_and(valid_out==-1,x==-1).sum()
fn=np.logical_and(valid_out==-1,x==1).sum()
fpr=fp/float(fp+tn)
tpr=tp/float(tp+fn)
print "%d %d %d %d" %(tp,fp,tn,fn)
print "FPR:%f,TPR:%f" %(fpr*100.0,tpr*100.0)
print "correct rate:%f" % (correct_ratio*100.0)
