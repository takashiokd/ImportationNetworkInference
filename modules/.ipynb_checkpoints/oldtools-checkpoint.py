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
from modules.LDS import lindyn_qp, Kalman_EM, update_A, LSWF


import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl


from datetime import date



from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


import inspect





# def prepare_HMM(counts, itermax, Nlins, filename, inpath, mcmc_options, mode='EM',Qplot='n'):
    
    
#     counts_deme= np.sum(counts,axis=1)
#     ND,slmax,tmax  = counts.shape
    
#     res_A_EM=[]
#     res_Ne_EM=[]
#     res_A_LS=[]
    
#     terminal_com=''
#     for iter in range(itermax):
#         setting = filename+'_iter'+str(iter)
#         aux= list(range(slmax))
#         random.shuffle(aux)
#         set_lins=np.array_split(aux ,Nlins)
#         counts_superlin=[]
#         for s in set_lins:
#             counts_superlin.append(np.sum(counts[:,s,:],axis=1))
#         counts_superlin=np.array(counts_superlin)
        
#         if iter==0:
#             aux=counts_superlin.flatten()
#             print("% of 0 components in B = ",round(100*len(aux[aux==0])/len(aux))," %")

#         counts_superlin+=1 #pseudocounts is used 
        
#         B = np.zeros((tmax,Nlins,ND))## ND: the number of age classes
#         for t in range(tmax):
#             for i in range(ND):
#                 if counts_deme[i,t]>0:
#                     B[t,:,i]= (counts_superlin[:,i,t])/counts_deme[i,t]
#                 else:
#                     B[t,:,i]= 0
                    
#         if iter==0:
#             print("B.shape",B.shape)
    
#         terminal_com+='./a.out -f '+setting+' '+mcmc_options

#         if iter==0 and Qplot=='y':
#             for i in range(Nlins):
#                 plt.plot(B[:,i,0])
#             plt.show()

#         A_LS=lindyn_qp(B, lam=0)

#         Ne_start=[1000]*ND

#         np.savetxt(inpath+'Aopt'+setting+'.csv', A_LS, fmt="%1.5f", delimiter=",")
#         np.savetxt(inpath+'countsdeme'+setting+'.csv', counts_deme, fmt="%d", delimiter=",")
#         Bshapelst=list(B.shape)
#         np.savetxt(inpath+'Bshape'+setting+'.csv', Bshapelst, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'Ne_start'+setting+'.csv', Ne_start, fmt="%1.5f", delimiter=",")

#         aux=B[:,0,:]
#         for i in range(1,Bshapelst[1]):
#             aux=np.concatenate((aux,B[:,i,:]),axis=0)
#         np.savetxt(inpath+'B'+setting+'.csv', aux, fmt="%1.5f", delimiter=",")  
    
#     return terminal_com, filename


# def prepare_HMM_repeat(counts, itermax, Nlins, num_repeat, filename, inpath,outpath, mcmc_options, mode='EM'):
    
    
#     counts_deme= np.sum(counts,axis=1)
#     ND,slmax,tmax  = counts.shape
#     print("ND,slmax,tmax ",ND,slmax,tmax )
    
#     res_A_EM=[]
#     res_Ne_EM=[]
#     res_A_LS=[]
    
#     terminal_com=''
#     for iter in range(itermax):
        
#         setting = filename+'_iter'+str(iter)
        
#         B =[]
#         for iter_repeat in range(num_repeat):
#             aux= list(range(slmax))
#             random.shuffle(aux)
#             set_lins=np.array_split(aux ,Nlins)
#             counts_superlin=[]
#             for s in set_lins:
#                 counts_superlin.append(np.sum(counts[:,s,:],axis=1))
#             counts_superlin=np.array(counts_superlin)
#             counts_superlin+=1  #pseudocounts is used 
#             # for t in range(tmax):
#             #     for i in range(ND):
#             #         for l in range(Nlins):
#             #             if counts_superlin[l,i,t]==0:
#             #                 counts_superlin[l,i,t]+=1
                            
            

#             Baux = np.zeros((tmax,Nlins,ND))## ND: the number of age classes

#             for t in range(tmax):
#                 for i in range(ND):
#                     if counts_deme[i,t]>0:
#                         Baux[t,:,i]= (counts_superlin[:,i,t])/counts_deme[i,t]
#                     else:
#                         Baux[t,:,i]= 0
                        
#             if iter_repeat==0:
#                 B = Baux.copy()
#             else:
#                 B = np.concatenate((B,Baux),axis=1)
#         B=np.array(B)
                
                    
#         if iter==0:
#             print("B.shape",B.shape)
    
#         terminal_com+='./a.out -f '+setting+' '+mcmc_options

#         if iter==0:
#             freqhist=B.flatten()
#             countzero=0
#             for i in freqhist:
#                 if i==0.0:
#                     countzero+=1
#             print("% of 0 components in B = ",round(100* countzero/len(freqhist))," %")

#             for i in range(Nlins):
#                 plt.plot(B[:,i,0])
#             plt.show()

#         A_LS=lindyn_qp(B, lam=0)

#         Ne_start=[1000]*ND

#         np.savetxt(inpath+'Aopt'+setting+'.csv', A_LS, fmt="%1.5f", delimiter=",")
#         np.savetxt(inpath+'countsdeme'+setting+'.csv', counts_deme, fmt="%d", delimiter=",")
#         Bshapelst=list(B.shape)
#         np.savetxt(inpath+'Bshape'+setting+'.csv', Bshapelst, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'Ne_start'+setting+'.csv', Ne_start, fmt="%1.5f", delimiter=",")

#         aux=B[:,0,:]
#         for i in range(1,Bshapelst[1]):
#             aux=np.concatenate((aux,B[:,i,:]),axis=0)
#         np.savetxt(inpath+'B'+setting+'.csv', aux, fmt="%1.5f", delimiter=",")  
    
#     return terminal_com, filename




# def infer_EM_LS_OLD(counts,EM_kws, Qplot='n'):
    
#     ##CHECK DATA
    
#     itermax=EM_kws['itermax']
#     Nlins=EM_kws['Nlins']
#     filename = EM_kws['filename']
#     outpath= EM_kws['outpath']
#     noisemode = EM_kws['noisemode']
        
            
#     ND,slmax,tmax  = counts.shape
#     print("ND,slmax,tmax ",ND,slmax,tmax )
    
#     B = construct_superfreq_nonpseudo(counts, Nlins)
#     aux=B.flatten()
#     print("{} % of components of non-pseudo B  are zero.".format(round(100* len(aux[aux==0.0])/len(aux))))

#     if Qplot=='y':
#         for i in range(Nlins):
#             plt.plot(B[:,i,0])
#         plt.show()
    
    
#     ##APPLY HMM 
    
#     res_A_EM=[]
#     res_Ne_EM=[]
#     res_A_LS=[]
#     res_A_LSWF=[]
#     res_Ne_LSWF =[]
    
#     counts_deme= np.sum(counts,axis=1)
#     counts_deme+=1
    
#     terminal_com=''
#     for iter in range(itermax):
#         setting = filename+'_iter'+str(iter)
        
# #         if noisemode==0:
# #             B = construct_superfreq(counts, Nlins)
# #         elif noisemode==1:
# #             B = construct_superfreq(counts, Nlins)
# #         elif noisemode==2:
# #             B = construct_superfreq(counts, Nlins)
            
#         B = construct_superfreq(counts, Nlins)
            
#         A_LS=lindyn_qp(B, lam=0)
        
#         A_LSWF, Ne_LSWF=LSWF(B)
        
        
#         lnLH_record, A_EM, Ne_EM=Kalman_EM(B,counts_deme,em_step_max=30, terminate_th=0.0001,noisemode=noisemode)
#         res_A_EM.append(A_EM)
#         res_Ne_EM.append(Ne_EM)
#         res_A_LS.append(A_LS)
#         res_A_LSWF.append(A_LSWF)
#         res_Ne_LSWF.append(Ne_LSWF)
        
        
#     res_A_EM = np.array(res_A_EM)   
#     res_Ne_EM = np.array(res_Ne_EM)
#     res_A_LS = np.array(res_A_LS)  
#     res_A_LSWF = np.array(res_A_LSWF) 
#     res_Ne_LSWF = np.array(res_Ne_LSWF)
    
    
#     np.save(outpath+ 'A_EM_'+filename+'.npy', res_A_EM)  
#     np.save(outpath+'Ne_EM_'+filename+'.npy', res_Ne_EM )  
#     np.save(outpath+'A_LS_'+filename+'.npy', res_A_LS) 
#     np.save(outpath+'A_LSWF_'+filename+'.npy', res_A_LSWF) 
#     np.save(outpath+'Ne_LSWF_'+filename+'.npy', res_Ne_LSWF) 
#     np.save(outpath+ 'countsdeme_'+filename+'.npy', counts_deme)
    
#     now = datetime.now()
#     with open(outpath+'log_'+filename+'.txt', 'w') as f:
#         f.write('filename, '+filename)
#         f.write('\n')
#         f.write('Data is created on '+now.strftime("%m/%d/%Y, %H:%M:%S"))
#         f.write('\n')
#         f.write('itermax = '+str(itermax))
#         f.write('\n')
#         f.write('ND,slmax,tmax'+str(ND)+','+str(slmax)+','+str(tmax))
#         f.write('\n')
#         f.write('Nlins= '+str(Nlins))
#         f.write('\n')
#         f.write('noisemode= '+str(noisemode))



def load_MCMC_old(dir_hmm, filename, itermax, burn_in =0.5,showplot=True):
    
    # if showplot==True:
    #     plt.figure(figsize=(10,3))
    for iter in range(itermax):

        setting = filename+'_iter'+str(iter)
        if iter==itermax-1:
            print(setting)
      
        A_mcmc=np.loadtxt(dir_hmm+'mcmc_A'+setting+'.csv',delimiter=',')
        #logLH_mcmc=np.loadtxt(dir_hmm+'mcmc_logLH'+setting+'.csv',delimiter=',')
        Ne_mcmc=np.loadtxt(dir_hmm+'mcmc_Ne'+setting+'.csv',delimiter=',')
        #para_mcmc=pd.read_csv(dir_hmm+ 'mcmc_para'+setting+'.csv',index_col=False,header=None)

        ND = len(Ne_mcmc[0])
        # if iter<3 and showplot==True:
            
            # if showplot==True:
            #     plt.plot(logLH_mcmc)

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
    # if showplot==True:
    #     plt.show()
    res_A_mcmc = np.array([ i.reshape((ND,ND)) for i in res_A_mcmc])

    return res_A_mcmc, res_Ne_mcmc