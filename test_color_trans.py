#!/usr/bin/python

import color_space_trans as ct

rgb=[100,230,120]

xyz=ct.RGB2XYZ(rgb)
lab=ct.XYZ2Lab(xyz)
print xyz
print lab
