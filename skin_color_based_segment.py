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
print valid_set_idx[0:10]
print tran_set_idx[0:10]

valid_set=data_set[valid_set_idx]
tran_set=data_set[tran_set_idx]

print valid_set.shape
print tran_set.shape

classifier=linear_model.LogisticRegression()
sample=tran_set[:,0:3]
sample=sample/256.0
target=tran_set[:,3]
target=(target-1.5)*-2.0
classifier.fit(sample,target)

test=valid_set[:,0:3]/256.0
valid_out=(valid_set[:,3]-1.5)*-2.0
x=classifier.predict(test)

corret_set=x==valid_out
print corret_set[0:10]
correct_ratio=(corret_set==True).sum()/float(valid_num)
#a=valid_out==1
#b=x==1
#print a[0:10]
#print b[0:10]
#print np.logical_and(a[0:10],b[0:10])
tp=np.logical_and(valid_out==1,x==1).sum()
fp=np.logical_and(valid_out==-1,x==1).sum()
tn=np.logical_and(valid_out==-1,x==-1).sum()
fn=np.logical_and(valid_out==-1,x==1).sum()

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

outfile=open('logistical_classifier.pkl','wb')
#cPickle.dump(classifier,outfile)
pickle.dump(classifier,outfile)
outfile.close()
