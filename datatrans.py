#!/usr/bin/python
import pickle
import random as rd

input_set_file=open("Skin_NonSkin.txt")

idx=0
input_set=[]
for line in input_set_file.readlines():
	tmp=[int(x) for x in line.split('\t')]
	input_set.append(tmp)

output=open('inputset.pkl','wb')
pickle.dump(input_set,output)
#rd.shuffle(input_set)
#pickle.dump(input_set[0:8000],output)
output.close()
