import os
import random
from statistics import harmonic_mean

import geopandas as gpd
from io import StringIO
from Bio import Phylo
import numpy as np
from numpy import linalg as LA
import time
import csv
from scipy.sparse import csc_matrix
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from pathlib import Path

from scipy import stats

from datetime import datetime
from datetime import timedelta

from modules.variables import CB_color_cycle
from modules.LDS import lindyn_qp, Kalman_EM, update_A, LSWF


import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl


from datetime import date



from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage



import multiprocess as mp

import inspect


import colorsys


def epiweek_date_v2(epiweek):
    return '['+str(epiweek)+']\n'+((datetime.strptime('2019-12-29', "%Y-%m-%d") + timedelta(days=7*(epiweek-1))).strftime('%b\n%d\n%Y'))



def min_max_scale(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x-min_val) / (max_val-min_val) for x in lst]


def calc_logit(f):
    f=np.array(f)
    return np.log(f/(1-f))


def calc_logit10(f):
    f=np.array(f)
    return np.log10(f/(1-f))

def calc_invlogit(psi):
    psi=np.array(psi)
    return np.exp(psi)/(1+np.exp(psi))

def generate_color_blind_friendly_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Distribute hues evenly
        hue_shifted = (hue + 0.5) % 1.0  # Shift the hue to create color-blind friendly palette
        rgb = colorsys.hsv_to_rgb(hue_shifted, 1, 1)  # Convert HSV to RGB
        colors.append(rgb)
    return colors

def calc_MSE(A,counts, totcounts):
    freq=calc_freq(counts,totcounts)
    MSE=[]
    for traj in range(freq.shape[1]):
        f=freq[:,traj,:]
        f1=f[:,:-1]
        f2=f[:,1:]
        MSE.append(np.mean(np.power(f2-A@f1,2))) 
    return np.mean(MSE)
    
def split_counts(counts, fraction):

    # shuffle the trajectory labels 
    b = counts.shape[1]
    indices = np.arange(b)
    np.random.shuffle(indices)
     # Split the indices into two parts
    b1 = int(b * fraction)
    indices1 = indices[:b1]
    indices2 = indices[b1:]
    # Split counts 
    counts1 = counts[:, indices1, :]
    counts2 = counts[:, indices2, :]

    return counts1, counts2

def reshape_data(X):
    X=np.array(X)
    new_shape = (X.shape[0] * X.shape[1],) + X.shape[2:]
    reshaped_array = np.reshape(X, new_shape)
    return reshaped_array


def Kdel(a,b):
    if a==b:
        res=1
    else:
        res=0
    return res

def calc_ridge_mat_intraregion(demelist,dict_deme_region):
    ND = len(demelist)
  
    Gam=np.zeros((ND,ND))
    for i in range(ND):
        re_i = dict_deme_region[demelist[i]]
        for j in range(ND):
            re_j = dict_deme_region[demelist[j]]
            if re_i ==re_j:
                Gam[i,j]=1
    # print(Gam)
    Lam=np.zeros((ND,ND,ND))
    for i in range(ND):
        for k in range(ND):
            re_k = dict_deme_region[demelist[k]]
            for l in range(ND):
                re_l = dict_deme_region[demelist[l]]
                if k!=i and l!=i:
                    for j in range(ND):
                        if j!=i:
                            Lam[i,k,l]+=Gam[j,k]*Kdel(l,k)+Gam[j,l]*Kdel(l,k)
                    Lam[i,k,l]-=2*Gam[k,l]
    return Lam/2


def calc_A_region_aved(A,regionnumlist):
    ND=len(regionnumlist)
    N_region=len(set(regionnumlist))
    A_region_aved=np.zeros((N_region,N_region))
    for re_i in range(N_region):
        fine_i  = np.where(np.array(regionnumlist)==re_i)[0]
        for re_j in range(N_region):
            fine_j = np.where(np.array(regionnumlist)==re_j)[0]
            aux=[]
            for i in fine_i:
                for j in fine_j:
                    if i!=j:
                        aux.append(A[i,j])
            if len(aux)>0:
                A_region_aved[re_i,re_j] = np.mean(aux)
            else:
                A_region_aved[re_i,re_j] = 10000000#This should not cause any problem

    A_fine_aved=np.zeros((ND,ND))
    for i in range(ND):
        for j in range(ND):
            A_fine_aved[i,j]=A_region_aved[regionnumlist[i],regionnumlist[j]]
    
    for i in range(ND):
        A_fine_aved[i,i]=0
        A_fine_aved[i,i]= 1- np.sum(A_fine_aved[i])
    
    return A_fine_aved
    
    
def calc_ridge_mat_diag(demelist):
    ND = len(demelist)

    Lam=np.zeros((ND,ND,ND))
    for i in range(ND):
        for k in range(ND):
            for l in range(ND):
                if k==l:
                    Lam[i,k,l]=1
        Lam[i,i,i]=0
        
    return Lam



def take_offdiag(A):
    aux =[]
    ND = len(A)
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                aux.append(A[i,j])
    return aux

def calc_errs(data):
    mean= np.mean(data,axis=0)
    aux = CI(data,alpha=0.5)
    return [mean-aux[0], aux[1]-mean]

def errscatter_Ax_Ay(res_Ax,res_Ay):
    if res_Ax.shape[1:] != res_Ay.shape[1:]:
        print('ERROR: matrices have different sizes')
    res_Ax_off = np.array([ take_offdiag(A) for A in res_Ax])
    res_Ay_off = np.array([ take_offdiag(A) for A in res_Ay])

    x = np.mean(res_Ax_off,axis=0)
    y = np.mean(res_Ay_off,axis=0)

    errs_x = np.array([calc_errs(res_Ax_off[:,i]) for i in range(res_Ax_off.shape[1])])
    errs_y = np.array([calc_errs(res_Ay_off[:,i]) for i in range(res_Ay_off.shape[1])])
    return x,errs_x, y,errs_y
    
    
def rm_diag(A):
    res=A.copy()
    for i in range(len(A)):
        res[i,i]=-1
    return res
    
def prt(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_str=[var_name for var_name, var_val in callers_local_vars if var_val is var][0]
    print(var_str+' = ',var)
    
def most_frequent(List):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    
    return max(set(List), key = List.count)

def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def calc_freq(counts,totcounts):
    freq= counts.copy()
    for i in range(counts.shape[0]):
        for t in range(counts.shape[2]):
            freq[i,:,t] *=1.0/totcounts[i,t]
    return freq

def CI(data,alpha):
    #alpha=0.1 returns [lowest 5%, largest 5%]
    sortdata=np.sort(data)
    return [sortdata[round(0.5*alpha*len(data))],sortdata[-round(0.5*alpha*len(data))]]

def CI_Ax_Ay(res_Ax,res_Ay):
    if res_Ax.shape[1:] != res_Ay.shape[1:]:
        print('ERROR: matrices have different sizes')
    res_Ax_off = np.array([ take_offdiag(A) for A in res_Ax])
    res_Ay_off = np.array([ take_offdiag(A) for A in res_Ay])

    x = np.mean(res_Ax_off,axis=0)
    y = np.mean(res_Ay_off,axis=0)

    CI_x = np.array([CI(res_Ax_off[:,i],alpha=0.5) for i in range(res_Ax_off.shape[1])])
    CI_y = np.array([CI(res_Ay_off[:,i],alpha=0.5) for i in range(res_Ay_off.shape[1])])
    
    return x,CI_x, y,CI_y


from scipy.linalg import null_space


from shapely.geometry import Polygon
from matplotlib.patheffects import withStroke


def take_offdiag(A):
    aux =[]
    ND = len(A)
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                aux.append(A[i,j])
    return np.array(aux)


def calc_effmig(gamma):
    ND = len(gamma)
    # gamma[j,i] denotes the transition prob (j -> i) 
    ns = null_space(gamma.transpose()-np.identity(len(gamma))) # Compute the steady-state vector
    A = np.zeros((ND,ND))
    for i in range(ND):
        for j in range(ND):
            A[i,j] = gamma[j,i]*ns[j]/ns[i] # normalize gamma by deme sizes
    return A





def Nei_Tajima_v2(freq_x,freq_y,samplesize):
    
    superlin=[]
    numtraj=1000
    trialmax=100000
    trial=0
    set_nonzero=[idx for idx,i in enumerate(freq_x) if i > 0]
    while len(superlin)<numtraj and trial<trialmax:
        trial+=1
        howmany_lins=random.sample(range(len(set_nonzero)), 1)[0]+1
        lins_selected=random.sample(set_nonzero, howmany_lins)
        suplin1=sum(freq_x[lins_selected])
        suplin2=sum(freq_y[lins_selected])
        if  suplin1>0.1 and  suplin1 <0.9:
            superlin.append([suplin1,suplin2])

    if len(superlin)>100:
        superlin=np.array(superlin)
        x=superlin[:,0]
        y=superlin[:,1]
        Fave=np.mean([np.power(x[i]-y[i],2)/( 0.5*(x[i]+y[i])*(1- 0.5*(x[i]+y[i]))  ) for i in range(len(superlin))])
        Neff=1/(Fave-2/samplesize)
    else:
        Neff=np.nan
    
    return  Neff



# # Tools for metadata/tree analysis



def epiweek_date(epiweek):
    return (datetime.strptime('2019-12-29', "%Y-%m-%d") + timedelta(days=7*(epiweek-1))).strftime("%Y/%m/%d")



def date_epiweek(d):
    d1= datetime.strptime(d, "%Y-%m-%d")
    d2 = datetime.strptime('2019-12-29', "%Y-%m-%d")

    return (d1 -d2).days//7+1

def ews_dates(ews):
    return [epiweek_date(i) for i in ews]

def rename_pango(pango):
   
    pango = str(pango)
    
    if pango =='nan':
        aux ='nan'
        
        
    elif pango=='B.1.1.7'or pango[0:2]=='Q.':
        aux='alpha'
        
    elif pango=='AY.4.2' or pango[0:7]=='AY.4.2.': # remove so-called delta plus
        aux='delta_plus'
    elif pango=='B.1.617.2' or pango[0:3]=='AY.':
        aux='delta'
        
            
    elif pango=='B.1.1.529' or pango=='BA.1':
        aux='omicron'
        
    elif pango[0:3]=='BA.':
        aux='omicron_other'
        
        
        
    elif pango=='B.1.177' or pango[0:8]=='B.1.177.':
        aux='B-1-177'
        
    elif pango=='B.1.351' or pango[0:8]=='B.1.351.':
        aux='beta'
        
    elif pango=='P.1' or pango[0:4]=='P.1.':
        aux='gamma'
        
    elif pango=='B.1.427' or pango=='B.1.429':
        aux='epsilon'    
        
    elif pango=='B.1.525':
        aux='eta'
    
    elif pango=='B.1.526':
        aux='iota'
        
    elif pango=='B.1.617.1':
        aux='kappa'
        
    elif pango=='B.1.617.1':
        aux='kappa'
    
    elif pango=='B.1.621' or pango==' B.1.621.1':
        aux='mu'
        
        
        
    else:
        aux='others'
    return aux

# a cllection of functions that may be useful
def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    if len(node_path)>=2:
        res=node_path[-2]
    else: 
        res="root"
    
    return res

def all_parents(tree):
    parents = {}
    for clade in tree.find_clades(order="level"):
        for child in clade:
            parents[child] = clade
    return parents

def lookup_by_names(tree):
    names = {}
    for clade in tree.find_clades():
        if clade.name:
            if clade.name in names:
                raise ValueError("Duplicate key: %s" % clade.name)
            names[clade.name] = clade
    return names

def tabulate_names(tree):
    names = {}
    for idx, clade in enumerate(tree.find_clades()):
        if clade.name:
            clade.name = "%d_%s" % (idx, clade.name)
        else:
            clade.name = str(idx)
        names[clade.name] = clade
    return names



# Added by TO
def remove_nodes(tree, listofnames):
    newtree = copy.deepcopy(tree)
    for name in listofnames:
        newtree.collapse(name)
    return newtree

def print_child(tree):
    for clade in tree.find_clades():
        print("parent = ",clade.name)
        for child in clade:
            print("         child = ",child)
            

def rm(strings):
    return strings[1:-1]


def to_int(lst):
    return [int(i) for i in lst]

def flatten(t):
    return [item for sublist in t for item in sublist]


'''
Returns the set of descendants of an internal node.
(eg. treestring='((L1L,L2L,L3L)I2I,L0L)I0I' and int_node='I2I')
'''
def get_descendants(treestring, int_node):
    pos_rangle=treestring.find(int_node)-1 # position of ) next to IxI
    counter=1
    shift=1
    # Move leftward until the number of ( becomes equal to the number of). 
    while counter>0:
        if treestring[pos_rangle-shift]==')':
            counter+=1
        elif treestring[pos_rangle-shift]=='(':
            counter-=1
        shift+=1
    pos_langle=pos_rangle-shift+1

    letters_inside=treestring[pos_langle:pos_rangle+1] 
    # Remark:the 2nd argument needed to be shifted by +1 in slicing
    Lpos=[pos for pos, char in enumerate(letters_inside) if char == 'L']
    numLL = int(len(Lpos)/2)
    externals_inside=[letters_inside[Lpos[2*i]:Lpos[2*i+1]+1] for i in range(numLL)]
    return externals_inside


def creat_hist(data):
    hist=Counter(data)
    list_elements=list(hist.keys())
    list_elements.sort()
    list_counts=[ hist[i] for i in list_elements]
    return [list_elements,list_counts]


def calc_PI_spectrum(res_A_mcmc):
    res_A_mcmc_rand = np.copy(res_A_mcmc)
        
    ND = int(len(res_A_mcmc[0]))
    res_PI_vec=[]
    res_spectrum=[]
    for i in range(len(res_A_mcmc_rand )):
        Amat = res_A_mcmc_rand[i]
        Leval, Levec=LA.eig(Amat.T)  
        idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
        Leval= Leval[idx] # Make sure the descending ordering
        Levec= Levec[:,idx]
        PI_vec= Levec[:,0] #Pick up the largest, corresponding to eigenvalue=1 (at least when the neutrality constraint is imposed)
        res_PI_vec.append(np.abs(PI_vec)/sum(abs(PI_vec)) )
        res_spectrum.append(np.abs(Leval))

    res_PI_vec=np.array(res_PI_vec)
    res_spectrum=np.array(res_spectrum)
    return res_PI_vec, res_spectrum


def calc_PI_eval(Amat):
    Leval, Levec=LA.eig(Amat.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]
    PI=np.abs(Levec[:,0])/sum(abs(Levec[:,0]))
    return np.array([PI, Leval])

# def calc_eval(Amat):
#     Leval, Levec=LA.eig(Amat.T)  
#     return Leval

def calc_res_PI_eval(res_A):
    num_workers = mp.cpu_count() 
    with mp.Pool(num_workers) as pool:
        res_PI_eval = pool.map(calc_PI_eval, res_A)
    return np.array(res_PI_eval)

# def calc_res_eval(res_A):
#     num_workers = mp.cpu_count() 
#     with mp.Pool(num_workers) as pool:
#         res_eval = pool.map(calc_eval, res_A)
#     return res_eval

def from_pos_to_Arc(pos,ND):
    return int(pos/ND),pos%ND

def from_Arc_to_pos(row,col,ND):
    return row*(ND)+col

def calc_A_mean_low_up(res_A_mcmc,alpha):
    ND = len(res_A_mcmc[0])
    A_mean=np.zeros((ND,ND))
    A_low=np.zeros((ND,ND))
    A_up=np.zeros((ND,ND))
    for row in range(ND):
        for col in range(ND):
            A_mean[row,col] = np.mean(res_A_mcmc[:,row,col])
            [lower,upper]=CI(res_A_mcmc[:,row,col],alpha=alpha) # alpha=0.5 corresponds to the upper/lower quartiles 
            A_low[row,col]=lower
            A_up[row,col]=upper
            
    return A_mean,A_low,A_up

def calc_A_med_low_up(res_A_mcmc,alpha):
    ND = len(res_A_mcmc[0])
    A_med=np.zeros((ND,ND))
    A_low=np.zeros((ND,ND))
    A_up=np.zeros((ND,ND))
    for row in range(ND):
        for col in range(ND):
            A_med[row,col] = np.median(res_A_mcmc[:,row,col])
            [lower,upper]=CI(res_A_mcmc[:,row,col],alpha=alpha) # alpha=0.5 corresponds to the upper/lower quartiles 
            A_low[row,col]=lower
            A_up[row,col]=upper

    return A_med,A_low,A_up

def calc_Ne_mean_low_up(res_Ne_mcmc,alpha):
    ND = len(res_Ne_mcmc[0])
    Ne_mean=np.zeros(ND)
    Ne_low=np.zeros(ND)
    Ne_up=np.zeros(ND)
    for i in range(ND):
            Ne_mean[i] = np.mean(res_Ne_mcmc[:,i])
            [lower,upper]=CI(res_Ne_mcmc[:,i],alpha=alpha) # alpha=0.5 corresponds to the upper/lower quartiles 
            Ne_low[i]=lower
            Ne_up[i]=upper
            
    return Ne_mean,Ne_low,Ne_up


def calc_gamma_mean_low_up(res_A_mcmc, demesize,alpha):
    ND = len(res_A_mcmc[0])
    gamma_mean=np.zeros((ND,ND))
    gamma_low=np.zeros((ND,ND))
    gamma_up=np.zeros((ND,ND))
    for row in range(ND):
        for col in range(ND):
            factor = demesize[row]/demesize[col]
            gamma_mean[row,col] = np.mean(res_A_mcmc[:,row,col])*factor
            [lower,upper]=CI(res_A_mcmc[:,row,col],alpha=alpha) # alpha=0.5 corresponds to the upper/lower quartiles 
            gamma_low[row,col]=lower*factor
            gamma_up[row,col]=upper*factor

    return gamma_mean,gamma_low,gamma_up



def make_txt_heatmap(mean, up, low, mode='with_err'):
    
    label=[]
    ND = len(mean)
    for i in range(ND):
        if mode =='with_err':
            aux=[str(np.round(mean[i,j],2))+'\n ['+str(np.round(low[i,j],2))
                 +', '+str(np.round(up[i,j],2))+']' for j in range(ND)]
        else:
            aux = [str(np.round(mean[i,j],2))  for j in range(ND)]
        label.append(aux)
    return np.array(label)


def make_txt_mat(mat):
    
    label=[]
    ND = len(mat)
    for i in range(ND):
        aux = [str(np.round(mat[i,j],2))  for j in range(ND)]
        label.append(aux)
    return np.array(label)


def calc_age_index(age_classes):
    
    index= [str(10*i[0])+' - '+str(10*(i[-1]+1)-1) for i in age_classes]
    return index




  
def maxoffdiag(mat):
    ND = len(mat)
    res=mat[0,1]
    for i in range(ND):
        for j in range(ND):
            if i!=j and res <mat[i,j]:
                res=mat[i,j]
    return res



#############################################################
# # Visualization
def curvline(start,end,rad,t=100,arrows=1,push=0.8):
    #Compute midpoint
    rad = rad/100.    
    x1, y1 = start
    x2, y2 = end
    y12 = (y1 + y2) / 2
    dy = (y2 - y1)
    cy = y12 + (rad) * dy
    #Prepare line
    tau = np.linspace(0,1,t)
    xsupport = np.linspace(x1,x2,t)
    ysupport = [(1-i)**2 * y1 + 2*(1-i)*i*cy + (i**2)*y2 for i in tau]
    #Create arrow data    
    arset = list(np.linspace(0,1,arrows+2))
    c = zip([xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)] for a in arset[1:-1]])
    dt = zip([xsupport[int(t*a*push)+1]-xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)+1]-ysupport[int(t*a*push)] for a in arset[1:-1]])
    arrowpath = zip(c,dt)
    return xsupport, ysupport, arrowpath

def plotcurv(start,end,rad,t=100,arrows=1,arwidth=.25,linewidth=1,color='black'):
    x, y, c = curvline(start,end,rad,t,arrows)
    plt.plot(x,y,'-',color=color,lw=linewidth)
    for d,dt in c:
        plt.arrow(d[0],d[1],dt[0],dt[1], shape='full', lw=0, 
                  length_includes_head=False, head_width=arwidth, color=color)
    return c

def draw_self_loop(ax, up_or_down, center, radius, rwidth=0.02, facecolor='#2693de', edgecolor='white', theta1=-30, theta2=180):
    
    # Add the ring
   
    ring = mpatches.Wedge(center, radius, theta1, theta2, width=rwidth)
    # Triangle edges
    offset = 0.05
    
    if up_or_down=='down':
    # Triangle edges
        xcent  = center[0] - radius + (rwidth/2)
        left   = [xcent - offset, center[1]]
        right  = [xcent + offset, center[1]]
        bottom = [(left[0]+right[0])/2., center[1]-0.15]
        arrow  = plt.Polygon([left, right, bottom, left])
    if up_or_down =='up':
    # Triangle edges
    
        xcent  = center[0] - radius + (rwidth/2)
        left   = [xcent - offset, center[1]]
        right  = [xcent + offset, center[1]]
        bottom = [(left[0]+right[0])/2., center[1]+0.15]
        arrow  = plt.Polygon([left, right, bottom, left])
        
    p = PatchCollection(
        [ring, arrow], 
        edgecolor = edgecolor, 
        facecolor = facecolor
    )
    ax.add_collection(p)







def calc_med_a(A):
    ND = len(A)
    aux=[]
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                aux.append(A[i,j])
    return np.round(np.median(aux),3)

def calc_large_a(A,frac):
    ND = len(A)
    aux=[]
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                aux.append(A[i,j])
    aux.sort()
    return np.round(aux[int(frac*(ND*ND-ND))],3)


def rotate(pos,angle, reflect='n'):
    pos_rot = []
    for i in pos:
        pos_rot.append([np.cos(2*np.pi*angle/360)*i[0] - np.sin(2*np.pi*angle/360)*i[1],
                        np.sin(2*np.pi*angle/360)*i[0] + np.cos(2*np.pi*angle/360)*i[1]])
    if reflect =='y':
        aux = []
        for i in pos_rot:
            aux.append([i[0],-1*i[1]])
        pos_rot = np.copy(aux)
        
    if reflect =='x':
        aux = []
        for i in pos_rot:
            aux.append([-1*i[0],i[1]])
        pos_rot = np.copy(aux)
    if reflect =='xy':
        aux = []
        for i in pos_rot:
            aux.append([-1*i[0],-1*i[1]])
        pos_rot = np.copy(aux)
        
    return np.array(pos_rot)





def array_join(arr, filename):
    return np.concatenate((arr,np.load(filename)),axis=0)




# def calc_Amean_err(res_A):
#     res_A =np.array(res_A)
#     ND = len(res_A[0])
#     lst_Amean=[]
#     lst_Astd=[]
#     #lst_Atrue=[]
#     for i in range(ND):
#         for j in range(ND):
#             if i!=j:
#                 lst_Amean.append(np.mean(res_A[:,i,j]))
#                 lst_Astd.append(np.std(res_A[:,i,j]))
#                 #lst_Atrue.append(A[i,j])
                
#     return lst_Amean, lst_Astd


################

def take_top_demes(df_lins, col_group,dict_deme_region,dict_region_number, gdf,ND):
    dict_deme_counts=Counter(df_lins[col_group])
    df_deme=pd.DataFrame({col_group: dict_deme_counts.keys() , 'counts' : dict_deme_counts.values() }).sort_values(by=['counts'],ascending=False).reset_index(drop=True)
    df_deme['counts'].plot()
    plt.xlabel('Rank of '+col_group)
    plt.ylabel('Nbr of sequences of the variant')
    plt.show()
    
    gdf=gdf.to_crs("epsg:3395") # 3395:Mercator Projection
    gdf["x"] = gdf.centroid.x
    gdf["y"] = gdf.centroid.y
    res=[]
    for i in df_deme[col_group]:
        aux=[]
        for j in gdf['NAME']:
            if i in j:
                 aux.append(j)
            if i=='Herefordshire, County of':
                if 'Herefordshire' in j:
                    aux.append(j)
            if i=='Essex':
                if 'Chelmsford' in j:
                    aux.append(j)

            if i=='Kent':
                if 'Maidstone' in j:
                    aux.append(j)
            if i =='Nottinghamshire':
                if 'Nottingham' in j:
                    aux.append(j)

            if i =='West Sussex':
                if 'Horsham' in j:
                    aux.append(j)
            if i=='Worcestershire':
                if 'Worcester' in j:
                    aux.append(j)
            if i=='North Yorkshire':
                 if 'Hambleton' in j:
                    aux.append(j)
            if i=='Bristol, City of':
                 if 'Bristol' in j:
                    aux.append(j)
            if i=='Cumbria':
                 if 'Eden' in j:
                    aux.append(j)
            if i=='East Sussex':
                 if 'Lewes' in j:
                    aux.append(j)
            if i=='Kingston upon Hull, City of':
                 if 'Kingston upon Hull' in j:
                    aux.append(j)

        if len(aux)>1:
            area=[]
            for a in aux:
                area.append(list(gdf[gdf['NAME']==a]['AREA'])[0])
            areamax=max(area)
            idx=area.index(areamax)
        else:
            idx=0
        if len(aux)>0:
            res.append(aux[idx])
        else:
            res.append(aux)
    df_deme['deme_shapefile']=res

    replace_byhand=[['Gloucestershire', 'Gloucester District (B)' ],
     ['Nottinghamshire','Mansfield District'],
     ['Lincolnshire',  'Lincoln District (B)' ],
      ['York', 'York (B)'] ,
     ['Bedford','Bedford (B)']
    ]

    for i in range(len(df_deme)):
        for j in range(len(replace_byhand)):
            if df_deme[col_group].iloc[i]==replace_byhand[j][0]:
                df_deme.loc[i,'deme_shapefile'] =replace_byhand[j][1]


    top_deme=list(df_deme[col_group].iloc[0:ND])

    #sort top_deme by regions according to dict_region_number
    deme_region=pd.DataFrame()
    deme_region['top_deme']=top_deme
    deme_region['region']=[ dict_deme_region[u] for u in deme_region['top_deme']]
    deme_region['region_number']=[dict_region_number[i] for i in deme_region['region']]
    deme_region=deme_region.sort_values(by=['region_number']).reset_index(drop=True)
    top_deme=list(deme_region['top_deme'])
    top_deme_region=list(deme_region['region'])

    aux =[]
    for i in range(len(top_deme)):
        demename_in_gdf = df_deme['deme_shapefile'].iloc[list(df_deme[col_group]).index(top_deme[i])]
        row=list(gdf['NAME']).index(demename_in_gdf)
        aux.append([gdf['x'].iloc[row],gdf['y'].iloc[row],gdf['geometry'].iloc[row]])
    df = pd.DataFrame()
    df['x'] = [ aux[i][0] for i in range(len(aux))]
    df['y'] = [ aux[i][1] for i in range(len(aux))]
    df['geometry'] = [ aux[i][2] for i in range(len(aux))]
    top_deme_gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    top_deme_gdf[col_group] = top_deme
    top_deme_gdf['region'] = top_deme_region
    return top_deme, top_deme_region, top_deme_gdf


def setting_utla(focal_variant):
    if focal_variant=='Delta':
        ND, Nlins=140, 50 #100,100 # Take the first ND utlas
        
        ew0=80
        width=18 
    elif focal_variant=='Alpha':
        ND=70 #, Take the first ND utlas
        Nlins=30
        ew0=55
        width=15

    elif focal_variant=='Omicron':
        ND=140 # Take the first ND utlas
        Nlins=50
        ew0=101
        width=9
    return dict({'ND':ND, 'Nlins':Nlins,'ew0':ew0,'width':width})


def setting_USstate(focal_variant):
    if focal_variant=='Delta':
        ND=51 # Take the first ND utlas
        Nlins=30#Nlins=30,60
        ew0=82
        width=15
    elif focal_variant=='Alpha':
        ND=51 #, Take the first ND utlas
        Nlins=25
        ew0=66
        width=12

    elif focal_variant=='Omicron':
        ND=51 # Take the first ND utlas
        Nlins=25
        ew0=104
        width=7
    return dict({'ND':ND, 'Nlins':Nlins,'ew0':ew0,'width':width})




##############################
def jump_dist_SIR(A, seqs, pops, dist, itermax=100000, mode ='S_i/S_j'):
    
    dist_norm = dist/np.max(dist)
    
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
    distlist=np.array(take_offdiag(dist_norm))

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
    

    return np.array(x)*np.max(dist), np.array(hist_dist)/np.max(dist)


def unifmat(ND):
    return np.ones((ND,ND))*1.0/ND


def calc_angle(xy):
    res=math.atan2(xy[1],xy[0])/math.pi*180
    if res <0:
        res+=360
    return res



# # Detailed balance
def db_ratio(Amat,Ath):
    #Compute Amat[i,j]/Amat[j,i], PI[j]/PI[i] for pairs with min[Aij,Aji]>Ath, where the order of (i,j) is defined s.t. PI[j]/PI[i]>1.
    
    ND = len(Amat)
    Leval, Levec=LA.eig(Amat.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]#location times rank

    if np.abs(Leval[0]-1.0)>0.01:
        print('ERROR: 0-th eigenvalue is not 1 but ',Leval[0])

    PI  = Levec[:,0].copy()
    if False in np.isreal(PI):
        print('ERRROR: PI is imaginary, something is wrong.')
    else:
        PI  = np.abs(PI)

    pairs=[]
    res =[]
    for i in range(ND):
        for j in range(i+1,ND):
            if min([Amat[i,j], Amat[j,i]])>Ath:
                if i<j:#PI[j]/PI[i]>Amat[i,j]/Amat[j,i]:
                    res.append([ Amat[i,j]/Amat[j,i], PI[j]/PI[i] ] )
                    pairs.append(str(j)+','+str(i))
                else:
                    res.append([ Amat[j,i]/Amat[i,j], PI[i]/PI[j] ] )
                    pairs.append(str(i)+','+str(j))
            
                    
    res = np.array(res)

    return res,pairs



##########################
# # Coarse-graining in England

def plot_dend(dist,labels):
    return dendrogram(linkage(squareform(dist), "complete"),color_threshold=np.max(squareform(dist))+1,labels=labels,orientation='right')
    
def clsters_cut_dend(dist,cut_threshold):
    mat = dist.copy()
    dists = squareform(mat)
    return fcluster(linkage(dists, method='complete'),cut_threshold,'distance')


def possible_clsters(dist,labels):
    dn=plot_dend(dist,labels)
    y=np.array([y[1] for y in dn['dcoord']])
    y.sort()
    
    df = pd.DataFrame({'label':labels})
    num_demes=[]
    for th in y:
        clusters=clsters_cut_dend(dist,cut_threshold=th)
        num_demes.append(len(set(clusters)))
        df['demes_per_cluster'+str(len(set(clusters)))] = [ i-1 for i in clusters]

    return df
  
    
    
#####################

def check_DB(A):
    Leval, Levec=LA.eig(A.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]
    PI_vec= Levec[:,0] #Pick up the largest, 
    PI_vec=np.abs(PI_vec)/sum(abs(PI_vec))
    ND = len(A)
    DB_violated=0
    for i in range(ND):
        for j in range(i+1,ND):
            DB_violated+=np.abs(PI_vec[i]*A[i,j]-PI_vec[j]*A[j,i])
            
    return DB_violated
            


def calc_flux_MMtilde(res_A,Tc):
    res_M=[]
    res_Mtilde=[]
    ND = len(res_A[0])
    for A in res_A:

        Reval, Revec=LA.eig(A)  
        indexed_complex_list = list(enumerate(Reval))
        # Sort by the real part, then by the imaginary part (ascending order) and keep the indices
        idx = [index for index, value in sorted(indexed_complex_list, key=lambda x: (x[1].real, x[1].imag),reverse=True)]
        Reval= Reval[idx].copy() # Make sure the descending ordering
        Revec= Revec[:,idx].copy()#location times rank
        normfac = Revec[0,0]
        for i in range(ND):
            Revec[:,i] *=1/normfac

        Leval, Levec=LA.eig(A.T)  
        indexed_complex_list = list(enumerate(Leval))
        # Sort by the real part, then by the imaginary part (ascending order) and keep the indices
        idx = [index for index, value in sorted(indexed_complex_list, key=lambda x: (x[1].real, x[1].imag),reverse=True)]
        Leval= Leval[idx] # Make sure the descending ordering
        Levec= Levec[:,idx]#location times rank

        for i in range(ND):
            Levec[:,i] *=1/np.dot(Revec[:,i], Levec[:,i])

        #Note: EVecs are orthonormal and leading left eigenvector is all ones.

        dd = np.linalg.matrix_power(A, Tc).copy()
        M = dd.copy()
        Mtilde = dd.copy()
        for i in range(ND):
            M[i,:] *= Levec[i,0].real
            Mtilde[:,i] *= 1.0/Levec[i,0].real

        res_M.append(M)
        res_Mtilde.append(Mtilde)

    res_M=np.array(res_M)
    res_Mtilde=np.array(res_Mtilde)
    
    return res_M, res_Mtilde


##########################





def calc_eig_Rvec_Lvec(Amat):
    ND = len(Amat)
    Reval, Revec=LA.eig(Amat)  
    
    indexed_complex_list = list(enumerate(Reval))
    # Sort by the real part, then by the imaginary part (ascending order) and keep the indices
    idx = [index for index, value in sorted(indexed_complex_list, key=lambda x: (x[1].real, x[1].imag),reverse=True)]
    Reval= Reval[idx] # Make sure the descending ordering
    Revec= Revec[:,idx]#location times rank
    normfac = Revec[0,0]
    for i in range(ND):
        Revec[:,i] *=1/normfac

    Leval, Levec=LA.eig(Amat.T)  
    indexed_complex_list = list(enumerate(Leval))
    # Sort by the real part, then by the imaginary part (ascending order) and keep the indices
    idx = [index for index, value in sorted(indexed_complex_list, key=lambda x: (x[1].real, x[1].imag),reverse=True)]
    
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]#location times rank

    for i in range(ND):
        Levec[:,i] *=1/np.dot(Revec[:,i], Levec[:,i])
    #Note: EVecs are orthonormal and leading left eigenvector is all ones.
        
    return Reval, Revec, Levec

# def calc_eig_Rvec_Lvec(Amat):
#     ND = len(Amat)
#     Reval, Revec=LA.eig(Amat)  
#     idx = np.abs(Reval).argsort()[::-1]  #The largest appears the left-most.
#     Reval= Reval[idx] # Make sure the descending ordering
#     Revec= Revec[:,idx]#location times rank
#     normfac = Revec[0,0]
#     for i in range(ND):
#         Revec[:,i] *=1/normfac

#     Leval, Levec=LA.eig(Amat.T)  
#     idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
#     Leval= Leval[idx] # Make sure the descending ordering
#     Levec= Levec[:,idx]#location times rank

#     for i in range(ND):
#         Levec[:,i] *=1/np.dot(Revec[:,i], Levec[:,i])
#     #Note: EVecs are orthonormal and leading left eigenvector is all ones.
        
#     return Reval, Revec, Levec


def compare_two_A(A_1,A_2):
    ND = len(A_1)
    evalue_1, rightevec_1, leghtevec_1 = calc_eig_Rvec_Lvec(A_1)
    evalue_2, rightevec_2, leghtevec_2 = calc_eig_Rvec_Lvec(A_2)

    v2A1v2=np.array([leghtevec_2[:,rank]@A_1@rightevec_2[:,rank]/(leghtevec_2[:,rank]@rightevec_2[:,rank]) for rank in range(ND)])
    v1A2v1=np.array([leghtevec_1[:,rank]@A_2@rightevec_1[:,rank]/(leghtevec_1[:,rank]@rightevec_1[:,rank]) for rank in range(ND)])
    
    return evalue_1,v1A2v1,evalue_2,v2A1v2


def make_annot(mat):
    
    mat_diag=mat.copy()
    for i in range(len(mat)):
        mat_diag[i,i]=0
        mat_diag[i,i]=1-np.sum(mat_diag[i])
    
    txt = []
    for i in range(len(mat)):
        aux=[]
        for j in range(len(mat)):
            aux.append(str(np.round(mat_diag[i,j],2)))
        txt.append(aux)
        
    return np.array(txt)



##########################

# def calc_ACG(res_A, res_PI, whichblock):
#     n_B = len(set(whichblock))
#     res_ACG =[]

#     for iter in range(len(res_A)):
#         A = res_A[iter].copy()
#         PI = res_PI[iter].copy()
#         ACG=np.zeros((n_B,n_B))
#         PIA = np.diag(PI) @ A
#         for I in range(n_B):
#             for J in range(n_B):
#                 i_in_I = np.where(whichblock==I)[0]
#                 j_in_J = np.where(whichblock==J)[0]
#                 ACG[I,J] =np.sum((PIA[i_in_I])[:,j_in_J])
#         for I in range(n_B):
#             ACG[I] *= 1.0/np.sum(ACG[I])
#         res_ACG.append(ACG.copy())
#     return np.array(res_ACG)
def calc_ACG(res_A, res_PI, whichblock):
    
    n_B = len(set(whichblock))
    whichblock = np.array(whichblock)
    res_ACG =[]
    if len(res_A[0])!=len(whichblock):
        print('ERROR:  len(A) != len(whichblock)')
    for iter in range(len(res_A)):
        A = res_A[iter].copy()
        PI = res_PI[iter].copy()
        ACG=np.zeros((n_B,n_B))
        PIA = np.diag(PI) @ A
        
        for I in range(n_B):
            i_in_I = np.where(whichblock==I)[0]
            rho_I = np.sum(PI[i_in_I])
            for J in range(n_B):
                if I!=J:
                    j_in_J = np.where(whichblock==J)[0]
                    ACG[I,J] =np.sum((PIA[i_in_I])[:,j_in_J])/rho_I
        for I in range(n_B):
            ACG[I,I] = 1- np.sum(ACG[I])
        res_ACG.append(ACG.copy())
    return np.array(res_ACG)



def calc_naiveACG(res_A,  whichblock):
    n_B = len(set(whichblock))
    res_ACG =[]

    for iter in range(len(res_A)):
        A = res_A[iter].copy()
        ACG=np.zeros((n_B,n_B))
        for I in range(n_B):
            for J in range(n_B):
                i_in_I = np.where(whichblock==I)[0]
                j_in_J = np.where(whichblock==J)[0]
                ACG[I,J] =np.sum((A[i_in_I])[:,j_in_J])
        for I in range(n_B):
            ACG[I] *= 1.0/np.sum(ACG[I])
        res_ACG.append(ACG.copy())
    return np.array(res_ACG)


def calc_PI(Amat):
    Leval, Levec=LA.eig(Amat.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]
    PI=np.abs(Levec[:,0])/sum(abs(Levec[:,0]))
    return PI


def calc_MFPT(A):
# ref: Charles Miller Grinstead, James Laurie Snell.
# For a given ergodic markov chain A_ij (\sum_j A_ij =1), compute the mean first passage time matrix (mij) to go grom i to j 
    n= len(A)

    w=calc_PI(A)
    W =np.zeros((n,n))
    for i in range(n):
        W[i] = w

    # Computet the Kemeny-Snell fundamental matrix Z
    Z = np.linalg.inv(np.identity(n) - A + W)

    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j] = (Z[j,j]-Z[i,j])/w[j]
    return M
######################
# # # selection

# def updatedensity_ode(xini,t0,t1,A,k, sigma):
#     ND = len(A)
    
#     dt =0.05
#     res=[]
#     x =np.copy(xini)
#     res.append(np.copy(x))
#     titermax =int( (t1-t0-1)/dt)
#     for titer in range(titermax):
#         t = t0+dt*(titer+1)
#         x_aux = np.copy(x)
#         for i in range(ND):
#             x[i] += k*x_aux[i]*dt
#             for j in range(ND):
#                 if j!=i:
#                     x[i] += (1+sigma)*A[i,j]*x_aux[j]*dt
#         if int(t)==t:          
#             res.append(np.copy(x))
#     return np.array(res)



                
# def updatefreq_ode(freqini,t0,t1,A,k, sigma):
#     dt =0.1
#     res=[]
#     freq =np.copy(freqini)
#     res.append(np.copy(freq))
#     titermax =int( (t1-t0-1)/dt)
#     ND = len(A)
#     for titer in range(titermax):
#         t = t0+dt*(titer+1)
#         freq_aux = np.copy(freq)
#         for i in range(ND):
#             freq[i] += k*freq_aux[i]*(1-freq_aux[i])*dt
#             for j in range(ND):
#                 if j!=i:
#                     freq[i]  += (1+sigma)*(A[i,j]*freq_aux[j] -A[i,j] *freq_aux[i])*dt
                    
#         if int(t)==t:          
#             res.append(np.copy(freq))
#     return np.array(res)           

def updatefreq_ode(freqini, t0, t1, A, k, sigma):
    dt = 0.1
    titermax = int((t1 - t0 - 1) / dt)
    freq = np.copy(freqini)
    res = [np.copy(freqini)]

    for titer in range(titermax):
        t = t0 + dt * (titer + 1)
        growth_term = k * freq * (1 - freq) * dt

        # Updated interaction term
        interaction_term = (1 + sigma) * dt * (A @ freq - np.sum(A, axis=1) * freq)
        
        
        freq += growth_term + interaction_term

        if int(t) == t:
            res.append(np.copy(freq))

    return np.array(res)


def updatefreq_ode_v2(freqini, t0, t1, A, k, m):
    dt = 0.1
    titermax = int((t1 - t0 - 1) / dt)
    freq = np.copy(freqini)
    res = [np.copy(freqini)]

    for titer in range(titermax):
        t = t0 + dt * (titer + 1)
        growth_term = k * freq * (1 - freq) * dt

        # Updated interaction term
        interaction_term =  dt * (A @ freq - np.sum(A, axis=1) * freq)
        
        selmig_term = m*dt*(freq*(A@(1-freq)) - np.diag(A)*freq*(1-freq))
        
        freq += growth_term + interaction_term+selmig_term

        if int(t) == t:
            res.append(np.copy(freq))

    return np.array(res)



def updatefreq_sde(freqini,t0,t1,A,k, sigma,Neff):
    dt =0.05
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
            freq[i] += k*freq_aux[i]*(1-freq_aux[i])*dt
            for j in range(ND):
                if j!=i:
                    freq[i]  += (1+sigma)*(A[i,j]*freq_aux[j] -A[i,j] *freq_aux[i])*dt
                    
            freq[i] +=np.sqrt(freq[i]*(1-freq[i])*dt/Neff[i])*np.random.normal()
            if freq[i]<0:
                freq[i]=0
            if freq[i]>1:
                freq[i]=1
                
        if int(t)==t:          
            res.append(np.copy(freq))
    return np.array(res)           



# def fit_sweep( mode, focal_variant,index, x_actual, res_A, t0,t1, tfit0,tfit1, round_max=10,filename='demo',k_fixed='n',sigma_fixed ='n',outpath='fig/selection/'):
#     Path(outpath).mkdir(parents=True, exist_ok=True)
    
#     ND = len(res_A[0])
#     itermax = len(res_A)
    
#     err_min=100000000

#     nonzeromin =[]
#     aux = x_actual.flatten()
#     for i in aux:
#         if i>0:
#             nonzeromin.append(i)
#     nonzeromin =np.min(nonzeromin)
            
#     ini_old = np.copy(x_actual[tfit0])
    
    
#     for idx, i in enumerate(ini_old):
#         if i<nonzeromin:
#             ini_old[idx] =nonzeromin
            
#     k_old =0.8
#     sigma_old = -0.5
#     k_res=[]
#     sigma_res=[]
#     err_res=[]
    
#     for rd in range(round_max):
        
#         if k_fixed =='n':
#             k= k_old*np.exp(np.random.normal(0,0.05))
#         else:
#             k =k_fixed
        
#         ini = ini_old*[np.exp(np.random.normal(0,0.1)) for i in range(ND)]
      
#         if sigma_fixed =='n':
#             sigma = sigma_old + np.random.normal(0,0.05)
#             if sigma<-1:
#                 sigma=-1
#         else:
#             sigma = sigma_fixed
            
#         err=0

#         for iter in range(itermax):
#             A = np.copy(res_A[iter])
            
#             if mode=='density':
#                 x_predicted=updatedensity_ode(ini,tfit0,tfit1,A,k, sigma)
#                 err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2)/x_actual[tfit0:tfit1].flatten())
#             elif mode=='frequency':
#                 x_predicted = updatefreq_ode(ini,tfit0,tfit1,A,k, sigma)
#                 err_mag  =  np.array([i*(1-i) for i in x_actual[tfit0:tfit1].flatten() ])
#                 err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2)/err_mag )
                
#           #  err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2))
            
            

#         if err_min>err:
#             err_min=err
#             ini_old = np.copy(ini)
#             k_old = k
#             sigma_old = sigma

#         k_res.append(k_old)
#         err_res.append(err_min)
#         sigma_res.append(sigma_old)
        
#     plt.figure(figsize = (16,3))
#     plt.subplot(1, 3, 1)
#     plt.plot(err_res)
#     plt.ylabel('err')
    
#     plt.subplot(1, 3, 2)
#     plt.plot(sigma_res)
#     plt.ylabel('sigma')
    
#     plt.subplot(1, 3, 3)
#     plt.plot(k_res)
#     plt.ylabel('k')
#     plt.show()
        
#     ini_opt = np.copy(ini_old)
#     k_opt = np.round(k_old,2)
#     sigma_opt = np.round(sigma_old,2)
    
    
#     t_predict=t1-(t0+tfit0)
#     trajs_opt=[]
#     for iter in range(itermax):
#         A = np.copy(res_A[iter])
#         if mode=='density':
#             x_predicted=updatedensity_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
#         elif mode=='frequency':
#             x_predicted=updatefreq_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
#         trajs_opt.append(x_predicted)
#     trajs_opt = np.copy(np.array(trajs_opt))

    
#     for yscale_mode in ['linear','log']:
    
#         for i in range(ND):
#             plt.scatter(range(t0,t1),x_actual[:,i],color=CB_color_cycle[i],label=index[i])
#             plt.plot(range(t0,t1),x_actual[:,i],'-',color=CB_color_cycle[i],alpha=0.2)
#             plt.plot(range(t0+tfit0,t0+tfit1),np.mean(trajs_opt,axis=0)[:tfit1-tfit0,i],'--',color=CB_color_cycle[i])
#             #plt.plot(range(t0+tfit1-1,t0+tfit0+t_predict),np.mean(trajs_opt,axis=0)[tfit1-tfit0-1:,i],'--',color=CB_color_cycle[i],alpha=0.3)
#         plt.ylim(np.min(x_actual),np.max(x_actual)*1.2)

#         plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='fit')
#         #plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='prediction')
#         plt.title('k='+str(k_opt)+', sigma='+str(sigma_opt)+'\n '+str(tfit1-tfit0)+' timepoints are used in fitting')
#         plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0))
        
#         if yscale_mode=='log':
#             plt.yscale('log')
#             if mode=='density':
#                 plt.ylabel('Density of '+focal_variant)
#                 plt.savefig(outpath+'log_densityspace'+filename+'.pdf',bbox_inches='tight')
#             elif mode=='frequency':
#                 plt.ylabel('Frequency of '+focal_variant)
#                 plt.savefig(outpath+'log_freqspace'+filename+'.pdf',bbox_inches='tight')
#             plt.show()
#         else:
#             if mode=='density':
#                 plt.ylabel('Density of '+focal_variant)
#                 plt.savefig(outpath+'densityspace'+filename+'.pdf',bbox_inches='tight')
#             elif mode=='frequency':
#                 plt.ylabel('Frequency of '+focal_variant)
#                 plt.savefig(outpath+'freqspace'+filename+'.pdf',bbox_inches='tight')
#             plt.show()
#############################################################


##############################



def calc_hist(data, xmin,xmax, numpoints, logmode='y'):
    data = np.array(data)
    xmin -=1e-10
    xmax +=1e-10
    if logmode=='n':
        x = np.linspace(xmin,xmax,numpoints)
    elif logmode=='y':
        x = np.exp(np.linspace(np.log(xmin),np.log(xmax),numpoints))
        
    xmed=[]
    density=[]
    for i in range(numpoints-1):
        counts = len(np.where((data>=x[i])& (data<x[i+1]))[0])
        xmed.append(0.5*x[i] + 0.5*x[i+1])
        dx = x[i+1]-x[i]
        density.append(counts/(len(data)*dx))
    density =np.array(density).astype(float)
        
    return np.array([xmed,density])

def random_jumps(Anormed,dist, itermax=100000, include_diag='n'):
    
    dist_norm = dist#/np.max(dist)
    
    IA = np.copy(Anormed)
    ND = len(Anormed)
    


    if include_diag=='n':
        problist=take_offdiag(IA)/np.sum(take_offdiag(IA))
        distlist=np.array(take_offdiag(dist_norm))
    elif  include_diag=='y':
        problist=IA.flatten()/np.sum(IA.flatten())
        distlist=np.array(dist_norm).flatten()
        
    pos_realized=np.random.choice(range(len(problist)), itermax, p = problist, replace=True)
    dist_realized = distlist[pos_realized]
    dist_realized=np.array([d*(1+np.random.normal(0,0.05)) for d in dist_realized])
    
    if len(Anormed)!=len(dist):
        print('error')
        dist_realized=[]
    
    return dist_realized


def random_jumps_SIR(A, seqs, pops, dist, itermax=100000, mode ='S_i/S_j',include_diag='n'):
    
    dist_norm = dist
    
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
            
    

    if include_diag=='n':
        problist=take_offdiag(IA)/np.sum(take_offdiag(IA))
        distlist=np.array(take_offdiag(dist_norm))
    elif  include_diag=='y':
        problist=IA.flatten()/np.sum(IA.flatten())
        distlist=np.array(dist_norm).flatten()
        
        
    pos_realized=np.random.choice(range(len(problist)), itermax, p = problist, replace=True)
    dist_realized = distlist[pos_realized]
    dist_realized=np.array([d*(1+np.random.normal(0,0.05)) for d in dist_realized])

    return dist_realized



def fit_powerlaw(x,y,xfitmin,xfitmax):
    fit_ind = np.where( (np.array(x)>xfitmin) & (np.array(x)<xfitmax) )[0]
    
    slope, intercept, r, p, std_err = stats.linregress(np.log(x[fit_ind[0]:fit_ind[-1]]), np.log(y[fit_ind[0]:fit_ind[-1]]))
    xth = x[fit_ind[0]:fit_ind[-1]]
    yth = [ np.exp(intercept)* np.exp(np.log(i)*slope) for i in xth]
    return np.round(slope,3), np.round(intercept,3), xth, yth

def levy_flight(N, alpha):
    
    m=1  # mode
    s = (np.random.pareto(alpha, N) + 1) * m

    """Simulates a 2D Lévy flight with N steps and characteristic exponent alpha."""
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    for i in range(1, N+1):
        theta = 2 * np.pi * np.random.rand()
        x[i] = x[i-1] + s[i-1] * np.cos(theta)
        y[i] = y[i-1] + s[i-1] * np.sin(theta)
    return x, y


def onedim_levy_flight(N, alpha):
    
    m=1  # mode
    s = (np.random.pareto(alpha, N) + 1) * m

    """Simulates a 2D Lévy flight with N steps and characteristic exponent alpha."""
    x = np.zeros(N+1)
    for i in range(1, N+1):
        theta = np.random.choice([-1,1])
        x[i] = x[i-1] + s[i-1] * theta
        
    return x

def onedim_levy_flight_box(N, alpha,L):
    
    m=1  # mode
    s = (np.random.pareto(alpha, N) + 1) * m

    """Simulates a 2D Lévy flight with N steps and characteristic exponent alpha."""
    x = np.zeros(N+1)
    for i in range(1, N+1):
        theta = np.random.choice([-1,1])
        x[i] = x[i-1] + s[i-1] * theta
        if x[i]>L:
            x[i] = L - (x[i]-L)
        elif x[i]<-L:
            x[i] = -L + (-L-x[i])
            
    return x



def A_SIR(A, seqs, pops, mode ='S_i/S_j'):
    

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
                
    return IA


###############
def compute_correlation(x, y, method='pearson'):
    from scipy.stats import pearsonr, spearmanr
    if method == 'pearson':
        correlation_coeff, p_value = pearsonr(x, y)
    elif method == 'spearman':
        correlation_coeff, p_value = spearmanr(x, y)
    else:
        raise ValueError("Invalid correlation method. Available options: 'pearson', 'spearman'.")
    return correlation_coeff
def calc_corr(res_A,Aref,method='pearson',th=0.01):
    corr=[]
    for A in res_A:
        aux=np.array(take_offdiag(A))
        aux_ref=np.array(take_offdiag(Aref))
        idx_clear=np.where(aux_ref>th)
        corr.append(compute_correlation(x=aux[idx_clear], y=aux_ref[idx_clear], method=method))
    return np.array(corr)




###############
def map_value_to_color(value, vmin=-3, vmax=3, cmap_name='rainbow'):
    """
    Maps a value in the range [vmin, vmax] to a color based on the given colormap name.

    Parameters:
    - value (float): The value to be mapped to a color.
    - vmin (float): The minimum value of the range.
    - vmax (float): The maximum value of the range.
    - cmap_name (str): The name of the colormap to use.

    Returns:
    - color (tuple): A tuple representing the RGB color.
    """
    
    # Clamp value within range
    value = max(vmin, min(value, vmax))
    
    # Normalize the value to the range [0, 1]
    normalized_value = (value - vmin) / (vmax - vmin)
    
    # Fetch the colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Get the RGB color
    color = cmap(normalized_value)
    
    return color




import scipy.stats as stats

def linear_fit_with_ci(x, y, confidence=0.95, intercept_zero=False):
    if intercept_zero:
        # Fit the data to a linear model with b = 0 (y = ax)
        a = np.sum(x * y) / np.sum(x**2)
        b = 0
        
        # Residual standard error
        y_fit = a * x
        residuals = y - y_fit
        n = len(x)
        residual_std_error = np.sqrt(np.sum(residuals**2) / (n - 1))
        
        # Standard error of a
        se_a = residual_std_error / np.sqrt(np.sum(x**2))
        
        # Calculate the t-critical value for the confidence level
        t_critical = stats.t.ppf((1 + confidence) / 2., n - 1)
        
        # Calculate the confidence interval for a
        a_ci = [a - t_critical * se_a, a + t_critical * se_a]
        b_ci = [0, 0]
    else:
        # Fit the data to a linear model y = ax + b
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        S_xx = np.sum((x - x_mean)**2)
        S_xy = np.sum((x - x_mean) * (y - y_mean))
        
        a = S_xy / S_xx
        b = y_mean - a * x_mean
        
        # Residual standard error
        y_fit = a * x + b
        residuals = y - y_fit
        residual_std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        # Standard errors of a and b
        se_a = residual_std_error / np.sqrt(S_xx)
        se_b = residual_std_error * np.sqrt(1 / n + x_mean**2 / S_xx)
        
        # Calculate the t-critical value for the confidence level
        t_critical = stats.t.ppf((1 + confidence) / 2., n - 2)
        
        # Calculate the confidence intervals
        a_ci = [a - t_critical * se_a, a + t_critical * se_a]
        b_ci = [b - t_critical * se_b, b + t_critical * se_b]
    
    return a, b, a_ci, b_ci
