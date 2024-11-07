import os
import random
from statistics import harmonic_mean

import geopandas as gpd
from io import StringIO
from Bio import Phylo
import numpy as np
from numpy import linalg as LA
import time
import copy
import csv
from scipy.sparse import csc_matrix
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from pathlib import Path


def jump_dist_SIR(A, seqs, pops, dist, itermax=100000, mode ='S_i/S_j'):
    
    if len(A)!=len(seqs):
        print('error')
    if len(A)!=len(dist):
        print('error')
        
    IA = np.copy(A)
    ND = len(A)
    
    for i in range(ND):
        for j in range(ND):
            if mode =='S_i/S_j':
                IA[i,j]*=seqs[i]/seqs[j]
            if mode =='S_i':
                IA[i,j]*=seqs[i]    
            if mode =='1/S_j':
                IA[i,j]*=1/seqs[j]
            if mode =='1':
                IA[i,j]*=1   
                
            if mode =='M_i/M_j':
                IA[i,j]*=pops[i]/pops[j]
            if mode =='M_i':
                IA[i,j]*=pops[i]    
            if mode =='1/M_j':
                IA[i,j]*=1/pops[j]
                
            if mode =='1/I_j':
                IA[i,j]*=1/seqs[j]
                
            if mode =='I_i/(M_i I_j)':
                IA[i,j]*=seqs[i]/(pops[i]*seqs[j])
            
    
    problist=take_offdiag(IA)/np.sum(take_offdiag(IA))
    distlist=np.array(take_offdiag(dist))

    pos_realized=np.random.choice(range(len(problist)), itermax, p = problist, replace=True)
    dist_realized = distlist[pos_realized]


    bins=500
    delta=1/500
    hist_dist=np.zeros(bins)

    for d in dist_realized:
        aux=d*(1+np.random.normal(0,0.1))
        if aux>1:
            aux=1
        hist_dist[round(np.ceil((aux-1)/delta))-1]+=1

    x = [delta*(i+1) for i in range(bins)]
    hist_dist*=1.0/(delta*itermax)
    

    return x, hist_dist