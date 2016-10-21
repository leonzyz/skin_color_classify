#!/usr/bin/python
import numpy as np


def RGB2Lab_vect(rgb_vect):
	lab_vect=np.array(rgb_vect,dtype=np.float32)
	return lab_vect
	'''
	lab_vect=np.array(rgb_vect,dtype=np.float32)
	idx=0
	for sample in rgb_vect:
		lab=RGB2Lab(sample)
		lab_vect[idx]=lab
		idx+=1

	return lab_vect
	'''

def RGB2Lab(rgb):
	xyz=RGB2XYZ(rgb)
	return XYZ2Lab(xyz)

def RGB2XYZ(rgb):
	R=rgb[0]
	G=rgb[1]
	B=rgb[2]
	var_R = ( R / 255.0 )        #R from 0 to 255
	var_G = ( G / 255.0 )        #G from 0 to 255
	var_B = ( B / 255.0 )        #B from 0 to 255

	if ( var_R > 0.04045 ):
		var_R = ( ( var_R + 0.055 ) / 1.055 ) ** 2.4
	else:                   
		var_R = var_R / 12.92
	if ( var_G > 0.04045 ):
		var_G = ( ( var_G + 0.055 ) / 1.055 ) ** 2.4
	else:
		var_G = var_G / 12.92
	if ( var_B > 0.04045 ):
		var_B = ( ( var_B + 0.055 ) / 1.055 ) ** 2.4
	else:
		var_B = var_B / 12.92

	var_R = var_R * 100.0
	var_G = var_G * 100.0
	var_B = var_B * 100.0

	# Observer. = 2 degree, Illuminant = D65
	X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
	Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
	Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
	return (X,Y,Z)

def XYZ2Lab(xyz):
	X=xyz[0]
	Y=xyz[1]
	Z=xyz[2]
	ref_X =  95.047   #Observer= 2 degree, Illuminant= D65
	ref_Y = 100.000
	ref_Z = 108.883
	var_X = X / ref_X          #ref_X =  95.047   Observer= 2 degree, Illuminant= D65
	var_Y = Y / ref_Y          #ref_Y = 100.000
	var_Z = Z / ref_Z          #ref_Z = 108.883

	if ( var_X > 0.008856 ):
		var_X = var_X ** ( 1/3.0 )
	else:
		var_X = ( 7.787 * var_X ) + ( 16 / 116.0 )
	if ( var_Y > 0.008856 ):
		var_Y = var_Y ** ( 1/3.0 )
	else:
		var_Y = ( 7.787 * var_Y ) + ( 16 / 116.0 )
	if ( var_Z > 0.008856 ):
		var_Z = var_Z ** ( 1/3.0 )
	else:
		var_Z = ( 7.787 * var_Z ) + ( 16 / 116.0 )

	L = ( 116.0 * var_Y ) - 16
	a = 500.0 * ( var_X - var_Y )
	b = 200.0 * ( var_Y - var_Z )
	return (L,a,b)

def Lab2XYZ(L,a,b):
	return (L,a,b)

def XYZ2RGB(x,y,z):
	return(x,y,z)

def RGB_norm(sample):
	sample_sum=np.sum(sample,dtype=np.float32,axis=1)
	sample_sum[sample_sum==0]=1
	divider=np.transpose(np.array([sample_sum,sample_sum,sample_sum]))
	return sample/divider

def RGB_norm2(sample):
	divider=256.0
	return sample/divider

def Lab_norm3(sample):
	sample_tmp=sample.copy()
	sample_tmp[0,:]=sample_tmp[0,:]-128
	return RGB_norm(sample_tmp)

def Lab_norm(sample):
	sample_sum=np.sum(sample[:,1:3],dtype=np.float32,axis=1)
	sample_sum[sample_sum==0]=1
	divider=np.transpose(np.array([sample_sum,sample_sum,sample_sum]))
	return sample/divider
	#return RGB_norm(sample)

def Lab_norm2(sample):
	return RGB_norm(sample)
