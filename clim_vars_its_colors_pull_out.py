# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:58:13 2020

@author: olga
"""
import numpy
import math
import scipy
import pandas   


c=[18,19,4,2,
5,15,10,8,
10,4,12,17,
19,2,15,7,
7,2,13,19,
19,18,1,14,
19,14,3,14,
13,9,14,5,17,17,2,16]

climvarscolors = ['yellow', #1
                  'violet',   #2
                  'violet',   #3
                  'violet',   #4
                  'yellow', #5     
                  'yellow', #6   
                  'violet',   #7
                  'yellow',  #8
                  'yellow',  #9  
                  'yellow',  #10 
                  'yellow',  #11
                  'royalblue1',    #12
                  'royalblue1',    #13
                  'royalblue1',    #14  
                  'royalblue1',    #15
                  'royalblue1',    #16
                  'royalblue1',    #17  
                  'royalblue1',    #18 
                  'royalblue1']    #19

v = numpy.array([]) 

for j in range(36):
    c[j]=c[j]-1

for j in range(36):
    print(j)
    v = numpy.append(v, climvarscolors[c[j]])
    
for j in range(36):
    print(v[j])    