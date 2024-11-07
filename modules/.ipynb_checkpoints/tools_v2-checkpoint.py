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


from datetime import datetime
from datetime import timedelta

from modules.variables import CB_color_cycle
from modules.LDS import lindyn_qp, Kalman_EM, update_A
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl



def most_frequent(List):
    return max(set(List), key = List.count)

def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def CI(data,alpha):
    sortdata=np.sort(data)
    return [sortdata[round(0.5*alpha*len(data))],sortdata[-round(0.5*alpha*len(data))]]

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
    return aux


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
    if len(res_A_mcmc)>5000:
        res_A_mcmc_rand = np.array(random.choices(res_A_mcmc, k=5000))
    else:
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

def calc_age_index(age_classes):
    
    index= [str(10*i[0])+' - '+str(10*(i[-1]+1)-1) for i in age_classes]
    return index


# # LOAD MCMC
def load_MCMC(dir_hmm, filename, itermax, burn_in =0.5):
    for iter in range(itermax):

        setting = filename+'_iter'+str(iter)
        if iter==itermax-1:
            print(setting)
      
        A_mcmc=np.loadtxt(dir_hmm+'mcmc_A'+setting+'.csv',delimiter=',')
        logLH_mcmc=np.loadtxt(dir_hmm+'mcmc_logLH'+setting+'.csv',delimiter=',')
        Ne_mcmc=np.loadtxt(dir_hmm+'mcmc_Ne'+setting+'.csv',delimiter=',')
        para_mcmc=pd.read_csv(dir_hmm+ 'mcmc_para'+setting+'.csv',index_col=False,header=None)

        ND = len(Ne_mcmc[0])
        if iter<3:
            plt.plot(logLH_mcmc)
            plt.show()
            # plt.plot(A_mcmc[:,0,1],label='A01')
            # plt.legend()
            # plt.show()
       
        mcmcsteps=len(A_mcmc)
   
        if iter==0:
            res_A_mcmc=np.copy(A_mcmc[int(burn_in*mcmcsteps):,:])
            res_Ne_mcmc=np.copy(Ne_mcmc[int(burn_in*mcmcsteps):,:])
        else:
            res_A_mcmc=np.concatenate((res_A_mcmc,A_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)
            res_Ne_mcmc=np.concatenate((res_Ne_mcmc,Ne_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)

    res_A_mcmc = np.array([ i.reshape((ND,ND)) for i in res_A_mcmc])

    return res_A_mcmc, res_Ne_mcmc


def load_MCMC_noplots(dir_hmm, filename, itermax, burn_in =0.5):
    for iter in range(itermax):

        setting = filename+'_iter'+str(iter)
        if iter==itermax-1:
            print(setting)
      
        A_mcmc=np.loadtxt(dir_hmm+'mcmc_A'+setting+'.csv',delimiter=',')
        logLH_mcmc=np.loadtxt(dir_hmm+'mcmc_logLH'+setting+'.csv',delimiter=',')
        Ne_mcmc=np.loadtxt(dir_hmm+'mcmc_Ne'+setting+'.csv',delimiter=',')
        para_mcmc=pd.read_csv(dir_hmm+ 'mcmc_para'+setting+'.csv',index_col=False,header=None)

        ND = len(Ne_mcmc[0])
        mcmcsteps=len(A_mcmc)
   
        if iter==0:
            res_A_mcmc=np.copy(A_mcmc[int(burn_in*mcmcsteps):,:])
            res_Ne_mcmc=np.copy(Ne_mcmc[int(burn_in*mcmcsteps):,:])
        else:
            res_A_mcmc=np.concatenate((res_A_mcmc,A_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)
            res_Ne_mcmc=np.concatenate((res_Ne_mcmc,Ne_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)

    res_A_mcmc = np.array([ i.reshape((ND,ND)) for i in res_A_mcmc])

    return res_A_mcmc, res_Ne_mcmc
  
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

    
    
def prepare_HMM(counts, itermax, Nlins, filename, inpath,outpath, mcmc_options, mode='EM'):
    counts_deme= np.sum(counts,axis=1)
    ND,slmax,tmax  = counts.shape
    print("ND,slmax,tmax ",ND,slmax,tmax )
    
    res_A_EM=[]
    res_Ne_EM=[]
    res_A_LS=[]
    
    terminal_com=''
    for iter in range(itermax):
        setting = filename+'_iter'+str(iter)
        aux= list(range(slmax))
        random.shuffle(aux)
        set_lins=np.array_split(aux ,Nlins)
        counts_superlin=[]
        for s in set_lins:
            counts_superlin.append(np.sum(counts[:,s,:],axis=1))
        counts_superlin=np.array(counts_superlin)

        B = np.zeros((tmax,Nlins,ND))## ND: the number of age classes
        
        for t in range(tmax):
            for i in range(ND):
                if counts_deme[i,t]>0:
                    B[t,:,i]= counts_superlin[:,i,t]/counts_deme[i,t]
                else:
                    B[t,:,i]= 0
        if iter==0:
            print("B.shape",B.shape)
    
        terminal_com+='./a.out -f '+setting+' '+mcmc_options

        if iter==0:
            freqhist=B.flatten()
            countzero=0
            for i in freqhist:
                if i==0.0:
                    countzero+=1
            print("% of 0 components in B = ",round(100* countzero/len(freqhist))," %")

            for i in range(Nlins):
                plt.plot(B[:,i,0])
            plt.show()

        A_LS=lindyn_qp(B, lam=0)

        Ne_start=[1000]*ND

        np.savetxt(inpath+'Aopt'+setting+'.csv', A_LS, fmt="%1.5f", delimiter=",")
        np.savetxt(inpath+'countsdeme'+setting+'.csv', counts_deme, fmt="%d", delimiter=",")
        Bshapelst=list(B.shape)
        np.savetxt(inpath+'Bshape'+setting+'.csv', Bshapelst, fmt="%d", delimiter=",")
        np.savetxt(inpath+'Ne_start'+setting+'.csv', Ne_start, fmt="%1.5f", delimiter=",")

        aux=B[:,0,:]
        for i in range(1,Nlins):
            aux=np.concatenate((aux,B[:,i,:]),axis=0)
        np.savetxt(inpath+'B'+setting+'.csv', aux, fmt="%1.5f", delimiter=",")  

#         if mode  ==  'EM':
#             lnLH_record, A_EM, Ne_EM=Kalman_EM(B,counts_deme,em_step_max=30, terminate_th=0.0001)
#             res_A_EM.append(A_EM)
#             res_Ne_EM.append(Ne_EM)
#             res_A_LS.append(A_LS)
#     if mode == 'EM':
#         res_A_EM = np.array(res_A_EM)   
#         res_Ne_EM = np.array(res_Ne_EM)
#         res_A_LS = np.array(res_A_LS)   
#         np.save(outpath+ 'A_EM_'+filename+'.npy', res_A_EM)  
#         np.save(outpath+'Ne_EM_'+filename+'.npy', res_Ne_EM )  
#         np.save(outpath+'A_LS_'+filename+'.npy', res_A_LS) 
    
    return terminal_com, filename


def construct_superfreq(counts, Nlins):
    counts_deme= np.sum(counts,axis=1)
    ND,slmax,tmax  = counts.shape
    
    aux= np.random.choice(slmax, slmax) # Bootstrapping: Sample lineages with replacement 
    set_lins=np.array_split(aux ,Nlins)
    counts_superlin=[]
    for s in set_lins:
        counts_superlin.append(np.sum(counts[:,s,:],axis=1))
    counts_superlin=np.array(counts_superlin)

    B = np.zeros((tmax,Nlins,ND))
    for t in range(tmax):
        for i in range(ND):
            if counts_deme[i,t]>0:
                B[t,:,i]= counts_superlin[:,i,t]/counts_deme[i,t]
            else:
                B[t,:,i]=0
    return B


def infer_EM_LS(counts, itermax, Nlins, filename, outpath ):
    counts_deme= np.sum(counts,axis=1)
    
    if np.min(counts_deme)==0:
        print('Counts of some deme is zere at some t')

        
    ND,slmax,tmax  = counts.shape
    print("ND,slmax,tmax ",ND,slmax,tmax )
    
    res_A_EM=[]
    res_Ne_EM=[]
    res_A_LS=[]
    
    terminal_com=''
    for iter in range(itermax):
        setting = filename+'_iter'+str(iter)
        
        B = construct_superfreq(counts, Nlins)
        # aux= np.random.choice(slmax, slmax) # Bootstrapping: Sample lineages with replacement 
        # set_lins=np.array_split(aux ,Nlins)
        # counts_superlin=[]
        # for s in set_lins:
        #     counts_superlin.append(np.sum(counts[:,s,:],axis=1))
        # counts_superlin=np.array(counts_superlin)

#         B = np.zeros((tmax,Nlins,ND))## ND: the number of age classes
        
#         for t in range(tmax):
#             for i in range(ND):
#                 if counts_deme[i,t]>0:
#                     B[t,:,i]= counts_superlin[:,i,t]/counts_deme[i,t]
#                 else:
#                     B[t,:,i]=0

        if iter==0:
            print("B.shape",B.shape)
    

        if iter==0:
            freqhist=B.flatten()
            countzero=0
            for i in freqhist:
                if i==0.0:
                    countzero+=1
            print("% of 0 components in B = ",round(100* countzero/len(freqhist))," %")

            for i in range(Nlins):
                plt.plot(B[:,i,0])
            plt.show()

        A_LS=lindyn_qp(B, lam=0)
        lnLH_record, A_EM, Ne_EM=Kalman_EM(B,counts_deme,em_step_max=30, terminate_th=0.0001)
        res_A_EM.append(A_EM)
        res_Ne_EM.append(Ne_EM)
        res_A_LS.append(A_LS)
        
    res_A_EM = np.array(res_A_EM)   
    res_Ne_EM = np.array(res_Ne_EM)
    res_A_LS = np.array(res_A_LS)   
    np.save(outpath+ 'A_EM_'+filename+'.npy', res_A_EM)  
    np.save(outpath+'Ne_EM_'+filename+'.npy', res_Ne_EM )  
    np.save(outpath+'A_LS_'+filename+'.npy', res_A_LS) 
    np.save(outpath+ 'countsdeme_'+filename+'.npy', counts_deme) 
    


def only_comm_HMM(itermax, filename, inpath,outpath, mcmc_options):

    terminal_com=''
    for iter in range(itermax):
        setting = filename+'_iter'+str(iter)

    
        terminal_com+='./a.out -f '+setting+' '+mcmc_options

    return terminal_com, filename    


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


def lineage_counter(df_lins, focal_variant, group, col_group,ew,width):
    variant_df = df_lins[df_lins['variant']==focal_variant][['epi_week',col_group,focal_variant]]
    variant_df=variant_df[variant_df[focal_variant].notnull()]
    variant_df=variant_df[variant_df[col_group].notnull()]
    
    
    list_lineages=list(set(variant_df[focal_variant]))
    ewmax=np.max(variant_df['epi_week'])+1

    dict_group_idx = dict()
    for idx, i in enumerate(group):
        dict_group_idx[i] =idx
    dict_lineage_idx = dict()
    for idx, i in enumerate(list_lineages):
        dict_lineage_idx[i] =idx

    counts = np.zeros((len(group),len(list_lineages),ewmax))
    for row in range(len(variant_df)):
        if variant_df[col_group].iloc[row] in group:
            counts[dict_group_idx[variant_df[col_group].iloc[row] ],dict_lineage_idx[variant_df[focal_variant].iloc[row]],variant_df['epi_week'].iloc[row]]+=1
    counts = counts[:,:,ew:ew+width]
    
    print('Number of sequences = ', np.sum(counts))
    return counts



def array_join(arr, filename):
    return np.concatenate((arr,np.load(filename)),axis=0)



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

# ######################
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
#     dt =0.05
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
