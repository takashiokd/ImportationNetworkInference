import os
import random
from statistics import harmonic_mean

import geopandas as gpd
from io import StringIO
# from Bio import Phylo
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

from datetime import datetime
from datetime import timedelta

from modules.variables import CB_color_cycle
from modules.LDS import *

import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl


from datetime import date


################

def LogLH_on_simplex(counts, Npop, noisemode,itermax, A0=None):

    ND = len(Npop)
    df = pd.DataFrame()
    for row in range(ND):
        res_A=[]
        res_LH=[]
        for iter in range(itermax):

            if A0 is None:
                A = Arand.copy()
            else:
                A = A0.copy()
            if iter>0:
                A[row] = np.random.dirichlet(([1]*ND),size=1).copy()[0]
            res_A.append(A.copy())
            res_LH.append(calc_LH(counts, A, Ne=Npop,noisemode=noisemode))
        
        df['LH'+str(row)]= res_LH
        df['A'+str(row)] = res_A
        #df=df.sort_values(by='LH',ascending=False).reset_index(drop=True)
    return df


def cutoff(a):
    res=a
    if a>1:
        res=1
    if a<0:
        res=0
    return res

def colfunc(val, minval, maxval, startcolor, stopcolor):
#     RED, YELLOW, GREEN  = (1, 0, 0), (1, 1, 0), (0, 1, 0)
#     CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    f = float(val-minval) / (maxval-minval)
    return tuple( cutoff(f*(b-a)+a) for (a, b) in zip(startcolor, stopcolor))