#!/usr/bin/python

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
