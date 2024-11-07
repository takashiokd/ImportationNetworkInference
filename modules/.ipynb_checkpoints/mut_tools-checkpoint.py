import os
import random
from statistics import harmonic_mean

# import geopandas as gpd
# from io import StringIO
# from Bio import Phylo
import numpy as np
from numpy import linalg as LA
import time
import copy
import csv

import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import Counter
from pathlib import Path

# from datetime import datetime
# from datetime import timedelta

import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl

import scipy

# from datetime import date

# from scipy.cluster.hierarchy import fcluster
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import dendrogram, linkage

# import inspect

from modules.HMMtools import *

import multiprocess as mp

###############################################
###############################################
# TOOLS FOR MUTATIONS
###############################################
###############################################
def bootstrap_over_mutations(counts):
    ND, Nmut, tmax = counts.shape
   
    counts_BS_aux = counts[:,np.random.choice(list(range(Nmut)),size = Nmut,replace = True),:]
    counts_BS  = counts_BS_aux.copy()

    return counts_BS


def calc_distAA_set(Q_carry_set):
    numAA = len(Q_carry_set)
    
    dist=np.zeros((numAA,numAA),dtype=np.float16)
    for i in range(numAA):
        for j in range(i+1,numAA):

            intersection = len(set.intersection(Q_carry_set[i],Q_carry_set[j]))    
            union = len(set.union(Q_carry_set[i], Q_carry_set[j]))  
            dist[i,j] = 1- intersection/union
            dist[j,i]=dist[i,j]
            
    return dist   


def adj_AA(dist,dist_th=0.5):
    adj = np.where(dist<dist_th, 1, 0)
    return adj
        
    

###############################################
###############################################
# TOOLS FOR MUTATIONS
###############################################
###############################################

    
def calc_mut_counts(args_mut,fth=0.15):
    country =args_mut['country']
    meta_col=args_mut['meta_col']
    demelist =args_mut['demelist']
    focal_variant =args_mut['focal_variant']
    ew =args_mut['ew']
    width =args_mut['width']
    dist_th=args_mut['dist_th']
    freqcut =args_mut['freqcut']
    itermax =args_mut['itermax']
    
    
    if 'Qsyn' in list(args_mut.keys()):
         Q_syn =args_mut['Qsyn']
    else:
         Q_syn = 'ERROR:Q_syn not specified'
    
    stg = '{}_{}_ew{}_width{}'.format(country,focal_variant,ew,width)
    print(stg)
    

    stg_dir = 'mutations/'+ Q_syn+'/'+stg+'/'
    print(Q_syn)
 
    ND = len(demelist)

    df_sampled = pd.read_csv(stg_dir+'sampled_AA.csv',index_col=0)
    dist= np.load(stg_dir+'dist_AA.npz')['arr_0'].copy()
    Q_carry = np.load(stg_dir+'Qcarry.npz')['arr_0'].copy() #sequences x activeAA
    meta_focal=pd.read_csv(stg_dir+'meta_focal.csv',index_col=0)
    #minmaxweek = np.load(stg_dir+'minmaxweek.npy')

    
    if meta_col=='block' and country =='England':# This is used for coarsegraining analysis of UTLAs.
        dict_u_block=args_mut['dict_u_block']
        blocklist=[]
        for u in meta_focal.utla:
            if u in dict_u_block.keys():
                blocklist.append(dict_u_block[u])
            else:
                blocklist.append(np.nan)
        meta_focal['block'] = blocklist
        
    if meta_col=='majoru' and country =='England':# This is used for coarsegraining analysis of UTLAs.
        dict_u_majoru=args_mut['dict_u_majoru']
        blocklist=[]
        for u in meta_focal.utla:
            if u in dict_u_majoru.keys():
                blocklist.append(dict_u_majoru[u])
            else:
                blocklist.append(np.nan)
        meta_focal['majoru'] = blocklist
        
        
    if meta_col == 'custom' and country =='England':
        custom_u_deme = args_mut['custom_u_deme']
        blocklist=[]
        for u in meta_focal.utla:
            if u in custom_u_deme.keys():
                blocklist.append(custom_u_deme[u])
            else:
                blocklist.append(np.nan)
        meta_focal['custom'] = blocklist
        
    if meta_col=='block' and country =='US':# This is used for coarsegraining analysis of US states.
        dict_u_block=args_mut['dict_u_block']
        blocklist=[]
        for u in meta_focal.state:
            if u in dict_u_block.keys():
                blocklist.append(dict_u_block[u])
            else:
                blocklist.append(np.nan)
        meta_focal['block'] = blocklist
        
        
    if meta_col=='block' and country =='Germany':# This is used for coarsegraining analysis of US states.
        dict_u_block=args_mut['dict_u_block']
        blocklist=[]
        for u in meta_focal.state:
            if u in dict_u_block.keys():
                blocklist.append(dict_u_block[u])
            else:
                blocklist.append(np.nan)
        meta_focal['block'] = blocklist
        
                
        
    num_active = len(dist)

    totcounts = np.zeros((ND,width))
    for i in range(len(meta_focal)):
        t=meta_focal['epi_week'].iloc[i]-ew
        deme =meta_focal[meta_col].iloc[i] 
        if deme in demelist:
             totcounts[demelist.index(deme),t]+=1
    totcounts+=1

    
    dict_demelist=dict()
    for idx,deme in enumerate(demelist):
        dict_demelist[deme]=idx
    counts_active =np.zeros((ND,num_active,width))
    def calc_counts_active(i):
        aux = np.where(Q_carry[:,i]==1)[0] #indices with that AA in metafocal
        df_aux = meta_focal.iloc[aux]

        res = np.zeros((ND, width))
        for j in range(len(df_aux)):
            t=df_aux['epi_week'].iloc[j]-ew
            deme = df_aux[meta_col].iloc[j] 
            if deme in demelist:
                res[dict_demelist[deme],t]+=1
        return res
    num_workers = mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        counts_active  = np.array(pool.map(calc_counts_active,list(range(num_active)))).transpose([1,0,2])

    #Extract major mutations
    major = find_major(counts_active,totcounts, freqcut,mode='mean',fth=fth)
        

    #Clustering major mutations
    dist_major  = dist[major].copy()
    dist_major =dist_major[:,major].copy()
    adj = adj_AA(dist_major,dist_th=dist_th)
    num_conn, which_conn=scipy.sparse.csgraph.connected_components(adj)   

    aux = [np.nan]*num_active # record the connected component for each major mutation
    for idx, m in enumerate(major):
        aux[m] = round(which_conn[idx])
    df_sampled['connected component'] = aux

    print('active mutations during ew {}-{} = '.format(ew,ew+width), num_active)
    print('major mutations  (freqcut={}) = {} '.format(freqcut,len(major)))
    print('#equivalent classes (dist_th={}) = {}'.format(dist_th,num_conn))    

    res_counts=[]
    
    new_columns = {}
    for iter in range(itermax):
        #From each connected componenet (equiv class), extract the mutation with the largest size as its representative 
        AA_rep_list=[] 
        # for cl in range(num_conn):
        #     AA_rep_list.append(major[np.random.choice(np.where(which_conn==cl)[0],size=1)[0]] )
        
        for cl in range(num_conn):
            mutation_indices = list(np.where(which_conn == cl)[0])
            mutation_counts = [np.sum(counts_active[:, major[i], :]) for i in mutation_indices]
            AA_rep_list.append(major[mutation_indices[np.argmax(mutation_counts)]])

        #Raw counts for representatives of major mutations
        counts_raw =counts_active[:,AA_rep_list,:]
        
        counts = counts_raw.copy()
        

        if iter==0:
            print('Zero elements: {} %'.format( round( len(np.where(counts_raw==0)[0])/(counts_raw.shape[0]*counts_raw.shape[1]*counts_raw.shape[2])*100,2)))   

        aux = [np.nan]*num_active
        for i in AA_rep_list:
            aux[i] = 1
        new_columns['iter{}'.format(iter)] = aux.copy()
        
        #df_sampled['iter{}'.format(iter)] = aux

        res_counts.append(counts.copy())
    
    df_sampled = pd.concat([df_sampled, pd.DataFrame(new_columns)], axis=1)
    
    return res_counts, totcounts, df_sampled


# def calc_mut_counts(args_mut):
#     country =args_mut['country']
#     meta_col=args_mut['meta_col']
#     demelist =args_mut['demelist']
#     focal_variant =args_mut['focal_variant']
#     ew =args_mut['ew']
#     width =args_mut['width']
#     dist_th=args_mut['dist_th']
#     freqcut =args_mut['freqcut']
#     itermax =args_mut['itermax']
    
    
#     if 'syn' in list(args_mut.keys()):
#          Q_syn =args_mut['syn']
#     else:
#          Q_syn = 'n'
    
    
#     stg = '{}_{}_ew{}_width{}'.format(country,focal_variant,ew,width)
#     print(stg)
    
#     if Q_syn=='y':
#         stg_dir = 'mutations/syn/'+stg+'/'
#         print('synonymous')
#     else:
#         stg_dir = 'mutations/all/'+stg+'/'
#         print('all')

#     ND = len(demelist)

#     df_sampled = pd.read_csv(stg_dir+'sampled_AA.csv',index_col=0)
#     dist= np.load(stg_dir+'dist_AA.npz')['arr_0'].copy()
#     Q_carry = np.load(stg_dir+'Qcarry.npz')['arr_0'].copy() #sequences x activeAA
#     meta_focal=pd.read_csv(stg_dir+'meta_focal.csv',index_col=0)
#     #minmaxweek = np.load(stg_dir+'minmaxweek.npy')

    
#     if meta_col=='block' and country =='England':# This is used for coarsegraining analysis of UTLAs.
#         dict_u_block=args_mut['dict_u_block']
#         blocklist=[]
#         for u in meta_focal.utla:
#             if u in dict_u_block.keys():
#                 blocklist.append(dict_u_block[u])
#             else:
#                 blocklist.append(np.nan)
#         meta_focal['block'] = blocklist
        
#     if meta_col=='majoru' and country =='England':# This is used for coarsegraining analysis of UTLAs.
#         dict_u_majoru=args_mut['dict_u_majoru']
#         blocklist=[]
#         for u in meta_focal.utla:
#             if u in dict_u_majoru.keys():
#                 blocklist.append(dict_u_majoru[u])
#             else:
#                 blocklist.append(np.nan)
#         meta_focal['majoru'] = blocklist
        
        
#     if meta_col == 'custom' and country =='England':
#         custom_u_deme = args_mut['custom_u_deme']
#         blocklist=[]
#         for u in meta_focal.utla:
#             if u in custom_u_deme.keys():
#                 blocklist.append(custom_u_deme[u])
#             else:
#                 blocklist.append(np.nan)
#         meta_focal['custom'] = blocklist
        
        
                
        
#     num_active = len(dist)

#     totcounts = np.zeros((ND,width))
#     for i in range(len(meta_focal)):
#         t=meta_focal['epi_week'].iloc[i]-ew
#         deme =meta_focal[meta_col].iloc[i] 
#         if deme in demelist:
#              totcounts[demelist.index(deme),t]+=1
#     totcounts+=1

    
#     dict_demelist=dict()
#     for idx,deme in enumerate(demelist):
#         dict_demelist[deme]=idx
#     counts_active =np.zeros((ND,num_active,width))
#     def calc_counts_active(i):
#         aux = np.where(Q_carry[:,i]==1)[0] #indices with that AA in metafocal
#         df_aux = meta_focal.iloc[aux]

#         res = np.zeros((ND, width))
#         for j in range(len(df_aux)):
#             t=df_aux['epi_week'].iloc[j]-ew
#             deme = df_aux[meta_col].iloc[j] 
#             if deme in demelist:
#                 res[dict_demelist[deme],t]+=1
#         return res
#     num_workers = mp.cpu_count()
#     with mp.Pool(num_workers) as pool:
#         counts_active  = np.array(pool.map(calc_counts_active,list(range(num_active)))).transpose([1,0,2])

#     #Extract major mutations
#     major = find_major(counts_active,totcounts, freqcut,mode='mean')
        

#     #Clustering major mutations
#     dist_major  = dist[major].copy()
#     dist_major =dist_major[:,major].copy()
#     adj = adj_AA(dist_major,dist_th=dist_th)
#     num_conn, which_conn=scipy.sparse.csgraph.connected_components(adj)   

#     aux = [np.nan]*num_active # record the connected component for each major mutation
#     for idx, m in enumerate(major):
#         aux[m] = round(which_conn[idx])
#     df_sampled['connected component'] = aux

#     print('active mutations during ew {}-{} = '.format(ew,ew+width), num_active)
#     print('major mutations  (freqcut={}) = {} '.format(freqcut,len(major)))
#     print('#equivalent classes (dist_th={}) = {}'.format(dist_th,num_conn))    

#     res_counts=[]
#     for iter in range(itermax):
#         #Extract a representative from each connected componenet
#         AA_rep_list=[] 
#         for cl in range(num_conn):
#             AA_rep_list.append(major[np.random.choice(list(np.where(which_conn==cl)[0]),size=1)[0]] )

#         #Raw counts for representatives of major mutations
#         counts_raw =counts_active[:,AA_rep_list,:]
        
#         counts = counts_raw.copy()
        

#         if iter==0:
#             print('Zero elements: {} %'.format( round( len(np.where(counts_raw==0)[0])/(counts_raw.shape[0]*counts_raw.shape[1]*counts_raw.shape[2])*100,2)))   

#         aux = [np.nan]*num_active
#         for i in AA_rep_list:
#             aux[i] = 1
#         df_sampled['iter{}'.format(iter)] = aux

#         res_counts.append(counts.copy())

#     return res_counts, totcounts, df_sampled





# # MUTATION HOKUSAI
# def M_prepare_EM_local(res_counts, M_mcmc_kws,df_sampled=None,Qplot='n'):
   
    
#     MCMC_dir = 'HMM_EM/'
    
#     dir_IO=M_mcmc_kws['dir_IO']
#     noisemodelist =M_mcmc_kws['noisemodelist']
#     mcmcsteps=M_mcmc_kws['mcmcsteps']
    
#     itermax=len(res_counts)
    
#     inpath=MCMC_dir+'input/'+dir_IO+'/'
#     Path(inpath).mkdir(parents=True, exist_ok=True)
    
#     ND,Nmut,tmax  = res_counts[0].shape
    
#     list_infilename =[]
    
#     if df_sampled is not None:
#         df_sampled.to_csv(inpath+'mutations_used.csv')
    
#     for iter, counts in enumerate(res_counts):
        
#         counts  = counts.transpose()
#         shapelst=list(counts.shape)
        
#         aux=counts[:,0,:].copy()
#         for i in range(1,Nmut):
#             aux=np.concatenate((aux,counts[:,i,:]),axis=0)
            
        
        
#         infilename = 'iter'+str(iter)
#         list_infilename.append(infilename)
#         np.savetxt(inpath+'counts_'+infilename+'.csv', aux, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'shape_'+infilename+'.csv', shapelst, fmt="%d", delimiter=",")
        
        
#     #sh files 
#     terminal_command=''
#     noisemodelist =mcmc_kws['noisemodelist']
#     mcmcsteps=mcmc_kws['mcmcsteps']
#     Path(MCMC_dir+'sh/'+dir_IO).mkdir(parents=True, exist_ok=True)
#     for iter in range(itermax):
#         for noisemode in noisemodelist:
#             for QDB in QDBlist:
                
#                 sh_filename=dir_IO+'/noisemode{}_{}_iter{}.sh'.format(noisemode,QDB,iter)
              
#                 text_file = open(MCMC_dir+'sh/'
#                                  +sh_filename, "w")
                
#                 outfilename = 'noisemode{}_{}_'.format(noisemode,QDB)+list_infilename[iter]
#                 mcmc_option = ' -f {} -g {} -d {} -m {} -n {} -D {}'.format(infilename, 
#                                                                             outfilename,
#                                                                       dir_IO, 
#                                                                       mcmcsteps,
#                                                                       noisemode ,
#                                                                       QDB)
#                 contents_sh ='./a.out'+mcmc_option+';'
#                 #write string to file
#                 text_file.write(contents_sh)
#                 #close file
#                 text_file.close()

#                 terminal_command+='sh sh/'+sh_filename+';'
        
#     return terminal_command



# def calc_mut_counts(args_mut):
#     country =args_mut['country']
#     meta_col=args_mut['meta_col']
#     demelist =args_mut['demelist']
#     focal_variant =args_mut['focal_variant']
#     ew =args_mut['ew']
#     width =args_mut['width']
#     dist_th=args_mut['dist_th']
#     freqcut =args_mut['freqcut']
#     itermax =args_mut['itermax']

    
#     stg = '{}_{}_ew{}_width{}'.format(country,focal_variant,ew,width)
#     print(stg)
#     stg_dir = 'mutations/corr/'+stg+'/'

#     ND = len(demelist)

#     df_sampled = pd.read_csv(stg_dir+'sampled_AA.csv',index_col=0)
#     dist= np.load(stg_dir+'dist_AA.npz')['arr_0'].copy()
#     Q_carry = np.load(stg_dir+'Qcarry.npz')['arr_0'].copy() #sequences x activeAA
#     meta_focal=pd.read_csv(stg_dir+'meta_focal.csv',index_col=0)
#     #minmaxweek = np.load(stg_dir+'minmaxweek.npy')


#     num_active = len(dist)

#     totcounts = np.zeros((ND,width))
#     for i in range(len(meta_focal)):
#         t=meta_focal['epi_week'].iloc[i]-ew
#         deme =meta_focal[meta_col].iloc[i] 
#         if deme in demelist:
#              totcounts[demelist.index(deme),t]+=1

#     #Compute the counts for active AA 
#     counts_active =np.zeros((ND,num_active,width))
#     for i_act in range(num_active):
#         aux = np.where(Q_carry[:,i_act]==1)[0] #indices with that AA in metafocal
#         df_aux = meta_focal.iloc[aux]
#         for i in range(len(df_aux)):
#             t=df_aux['epi_week'].iloc[i]-ew
#             deme = df_aux[meta_col].iloc[i] 
#             if deme in demelist:
#                  counts_active[demelist.index(deme),i_act,t]+=1

#     #Extract major mutations
#     inifreq=np.sum(counts_active,axis=0)[:,0]/np.sum(totcounts[:,0])
#     major = np.where(inifreq>freqcut)[0]

#     #Clustering major mutations
#     dist_major  = dist[major].copy()
#     dist_major =dist_major[:,major].copy()
#     adj = adj_AA(dist_major,dist_th=dist_th)
#     num_conn, which_conn=scipy.sparse.csgraph.connected_components(adj)   

#     aux = [np.nan]*num_active # record the connected component for each major mutation
#     for idx, m in enumerate(major):
#         aux[m] = round(which_conn[idx])
#     df_sampled['connected component'] = aux

#     print('active mutations during ew {}-{} = '.format(ew,ew+width), num_active)
#     print('major mutations  (freqcut={}) = {} '.format(freqcut,len(major)))
#     print('#equivalent classes (dist_th={}) = {}'.format(dist_th,num_conn))    

#     res_counts=[]
#     for iter in range(itermax):
#         #Extract a representative from each connected componenet
#         AA_rep_list=[] 
#         for cl in range(num_conn):
#             AA_rep_list.append(major[np.random.choice(list(np.where(which_conn==cl)[0]),size=1)[0]] )

#         #Raw counts for representatives of major mutations
#         counts_raw =counts_active[:,AA_rep_list,:]

#         # Both EM and MCMC take the lineage count data as input and the total sequences are computed by summing over lineage sequences.
#         # To use these scripts for the mutation data, normalize the mutation counts such that the sum over mutations agrees with the total sequences
#         counts = counts_raw.copy()
#         for i in range(ND):
#             for t in range(width):
#                 if np.sum(counts_raw[i,:,t])>0:
#                     counts[i,:,t]*=round(totcounts[i,t]/np.sum(counts_raw[i,:,t]),2)

#         if iter==0:
#             print('Zero elements: {} %'.format( round( len(np.where(counts_raw==0)[0])/(counts_raw.shape[0]*counts_raw.shape[1]*counts_raw.shape[2])*100,2)))   

#         aux = [np.nan]*num_active
#         for i in AA_rep_list:
#             aux[i] = 1
#         df_sampled['iter{}'.format(iter)] = aux

#         res_counts.append(counts.copy())

#     return res_counts, df_sampled


# ########################
# #functions that prepare MCMC input.
# #Use appropriate one
# #local computer or HOKUSUAI ?
# #lineage data or mutation data ? 

# # LINEAGE LOCAL
# def prepare_MCMC_local(counts,  mcmc_kws,Qplot='n'):
    
#     MCMC_dir = 'HMM_MCMC/'
    
#     itermax=mcmc_kws['itermax']
#     Nlins=mcmc_kws['Nlins']
#     dir_IO=mcmc_kws['dir_IO']

#     inpath=MCMC_dir+'input/'+dir_IO+'/'
#     Path(inpath).mkdir(parents=True, exist_ok=True)
    
#     counts_deme= np.sum(counts,axis=1)
#     ND,slmax,tmax  = counts.shape
    
#     list_infilename =[]
    
#     for iter in range(itermax):
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

#         counts_superlin =counts_superlin.transpose([2,0,1]) #Pseudo counts will be added later in cpp 
#         aux=counts_superlin[:,0,:].copy()
#         for i in range(1,Nlins):
#             aux=np.concatenate((aux,counts_superlin[:,i,:]),axis=0)
           
#         shapelst=list(counts_superlin.shape)
        
#         infilename = 'iter'+str(iter)
#         list_infilename.append(infilename)
#         np.savetxt(inpath+'counts_'+infilename+'.csv', aux, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'shape_'+infilename+'.csv', shapelst, fmt="%d", delimiter=",")
        
        
#     #sh files 
#     terminal_command=''
#     QDBlist=mcmc_kws['QDBlist']
#     noisemodelist =mcmc_kws['noisemodelist']
#     mcmcsteps=mcmc_kws['mcmcsteps']
#     Path(MCMC_dir+'sh/'+dir_IO).mkdir(parents=True, exist_ok=True)
#     for iter in range(itermax):
#         for noisemode in noisemodelist:
#             for QDB in QDBlist:
                
#                 sh_filename=dir_IO+'/noisemode{}_{}_iter{}.sh'.format(noisemode,QDB,iter)
              
#                 text_file = open(MCMC_dir+'sh/'
#                                  +sh_filename, "w")
                
#                 outfilename = 'noisemode{}_{}_'.format(noisemode,QDB)+list_infilename[iter]
#                 mcmc_option = ' -f {} -g {} -d {} -m {} -n {} -D {}'.format(list_infilename[iter], 
#                                                                             outfilename,
#                                                                       dir_IO, 
#                                                                       mcmcsteps,
#                                                                       noisemode ,
#                                                                       QDB)
#                 contents_sh ='./a.out'+mcmc_option+';'
#                 #write string to file
#                 text_file.write(contents_sh)
#                 #close file
#                 text_file.close()

#                 terminal_command+='sh sh/'+sh_filename+';'
        
#     return terminal_command

# # LINEAGE HOKUSAI
# def prepare_MCMC_HOKUSAI(counts, mcmc_kws,Qplot='n'):
    
#     MCMC_dir = 'HMM_MCMC/'
    
#     dir_IO=mcmc_kws['dir_IO']
#     itermax=mcmc_kws['itermax']
#     Nlins=mcmc_kws['Nlins']

    
    
#     inpath=MCMC_dir+'input/'+dir_IO+'/'
#     Path(inpath).mkdir(parents=True, exist_ok=True)
    
#     counts_deme= np.sum(counts,axis=1)
#     ND,slmax,tmax  = counts.shape
    
#     list_infilename =[]
    
#     for iter in range(itermax):
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

#         counts_superlin =counts_superlin.transpose([2,0,1]) #Pseudo counts will be added later in cpp 
        
#         aux=counts_superlin[:,0,:].copy()
#         for i in range(1,Nlins):
#             aux=np.concatenate((aux,counts_superlin[:,i,:]),axis=0)
           
#         shapelst=list(counts_superlin.shape)
        
#         infilename = 'iter'+str(iter)
#         list_infilename.append(infilename)
#         np.savetxt(inpath+'counts_'+infilename+'.csv', aux, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'shape_'+infilename+'.csv', shapelst, fmt="%d", delimiter=",")
        
        
#     #sh files 
#     terminal_command=''
#     QDBlist=mcmc_kws['QDBlist']
#     noisemodelist =mcmc_kws['noisemodelist']
#     mcmcsteps=mcmc_kws['mcmcsteps']
#     Path(MCMC_dir+'sh/'+dir_IO).mkdir(parents=True, exist_ok=True)
#     for iter in range(itermax):
#         for noisemode in noisemodelist:
#             for QDB in QDBlist:
                
#                 sh_filename=dir_IO+'/HOKUSAI_noisemode{}_{}_iter{}.sh'.format(noisemode,QDB,iter)
              

#                 outfilename = 'noisemode{}_{}_'.format(noisemode,QDB)+list_infilename[iter]
#                 mcmc_option = ' -f {} -g {} -d {} -m {} -n {} -D {}'.format(list_infilename[iter], 
#                                                                             outfilename,
#                                                                       dir_IO, 
#                                                                       mcmcsteps,
#                                                                       noisemode ,
#                                                                       QDB)
                
#                 contents_sh = '#!/bin/sh\n'
#                 contents_sh +='#------ pjsub option --------#\n'
#                 contents_sh +='#PJM -L rscunit=bwmpc\n'
#                 contents_sh +='#PJM -L elapse=24:00:00\n'
#                 contents_sh +='#PJM -L vnode=1\n'
#                 contents_sh +='#PJM -L vnode-core=2\n'
#                 #contents_sh +='#PJM -o cout/cout_'+dir_IO.replace('/','_')+'noisemode{}_{}_iter{}'.format(noisemode,QDB,iter)+'\n'
#                 contents_sh +='#------- Program execution ---#\n'

#                 contents_sh +='./a.out'+mcmc_option+';'
                
#                 text_file = open(MCMC_dir+'sh/'
#                  +sh_filename, "w")

#                 #write string to file
#                 text_file.write(contents_sh)
#                 #close file
#                 text_file.close()

#                 terminal_command+='pjsub sh/'+sh_filename+';'
    
#     return terminal_command


# # MUTATION HOKUSAI
# def M_prepare_MCMC_HOKUSAI(res_counts, M_mcmc_kws,df_sampled=None,Qplot='n'):
   
    
#     MCMC_dir = 'HMM_MCMC/'
    
#     dir_IO=M_mcmc_kws['dir_IO']
#     QDBlist=M_mcmc_kws['QDBlist']
#     noisemodelist =M_mcmc_kws['noisemodelist']
#     mcmcsteps=M_mcmc_kws['mcmcsteps']
    
#     itermax=len(res_counts)
    
#     inpath=MCMC_dir+'input/'+dir_IO+'/'
#     Path(inpath).mkdir(parents=True, exist_ok=True)
    
#     ND,Nmut,tmax  = res_counts[0].shape
    
#     list_infilename =[]
    
#     if df_sampled is not None:
#         df_sampled.to_csv(inpath+'mutations_used.csv')
    
#     for iter, counts in enumerate(res_counts):
        
#         counts  = counts.transpose()
#         shapelst=list(counts.shape)
        
#         aux=counts[:,0,:].copy()
#         for i in range(1,Nmut):
#             aux=np.concatenate((aux,counts[:,i,:]),axis=0)
            
        
        
#         infilename = 'iter'+str(iter)
#         list_infilename.append(infilename)
#         np.savetxt(inpath+'counts_'+infilename+'.csv', aux, fmt="%d", delimiter=",")
#         np.savetxt(inpath+'shape_'+infilename+'.csv', shapelst, fmt="%d", delimiter=",")
        
        
#     #sh files 
#     terminal_command=''

#     Path(MCMC_dir+'sh/'+dir_IO).mkdir(parents=True, exist_ok=True)
#     for iter in range(itermax):
#         for noisemode in noisemodelist:
#             for QDB in QDBlist:
                
#                 sh_filename=dir_IO+'/HOKUSAI_noisemode{}_{}_iter{}.sh'.format(noisemode,QDB,iter)
              

#                 outfilename = 'noisemode{}_{}_'.format(noisemode,QDB)+list_infilename[iter]
#                 mcmc_option = ' -f {} -g {} -d {} -m {} -n {} -D {}'.format(list_infilename[iter], 
#                                                                             outfilename,
#                                                                       dir_IO, 
#                                                                       mcmcsteps,
#                                                                       noisemode ,
#                                                                       QDB)
                
#                 contents_sh = '#!/bin/sh\n'
#                 contents_sh +='#------ pjsub option --------#\n'
#                 contents_sh +='#PJM -L rscunit=bwmpc\n'
#                 contents_sh +='#PJM -L elapse=24:00:00\n'
#                 contents_sh +='#PJM -L vnode=1\n'
#                 contents_sh +='#PJM -L vnode-core=2\n'
#                 #contents_sh +='#PJM -o cout/cout_'+dir_IO.replace('/','_')+'noisemode{}_{}_iter{}'.format(noisemode,QDB,iter)+'\n'
#                 contents_sh +='#------- Program execution ---#\n'

#                 contents_sh +='./a.out'+mcmc_option+';'
                
#                 text_file = open(MCMC_dir+'sh/'
#                  +sh_filename, "w")

#                 #write string to file
#                 text_file.write(contents_sh)
#                 #close file
#                 text_file.close()

#                 terminal_command+='pjsub sh/'+sh_filename+';'
    
#     return terminal_command