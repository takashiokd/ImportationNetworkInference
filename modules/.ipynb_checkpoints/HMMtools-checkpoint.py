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
from modules.LDS import lindyn_qp, Kalman_EM, update_A, LSWF

import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl


from datetime import date

# from scipy.cluster.hierarchy import fcluster
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import inspect


def find_major(counts,totcounts, freqcut, mode='mean',fth=0.15):
    
    T = totcounts.shape[1]
    if mode=='ini': 
        major=[]
        for i in range(counts.shape[1]):
            f=np.sum(counts[:,i,:],axis=0)/np.sum(totcounts,axis=0)
            freqini = f[0]
            freqend = f[-1]
            df=[ np.abs(f[t+1]-f[t]) for t in range(T-1)]
            
            if freqini>freqcut and freqini < fth and freqend>0 and freqend<1 and np.max(df)<0.1:
                major.append(i)
                
    if mode=='mean': 
        #find trajectories which exists at the initial and last timepoints and also 
        major=[]
        for i in range(counts.shape[1]):
            f=np.sum(counts[:,i,:],axis=0)/np.sum(totcounts,axis=0)
            freqini = f[0]
            freqend = f[-1]
            freq_mean = np.mean(f)
            df=[ np.abs(f[t+1]-f[t]) for t in range(T-1)]
       
            if np.sum(counts[:,i,0])>0 and np.sum(counts[:,i,-1])>0 and freq_mean>freqcut and freq_mean<fth and np.max(df)<0.1:
                major.append(i)
        
    return major






def counts_roll(counts, dt=2):
   
    width = len(counts[0,0,:])

    width_after = width//dt
    counts_after = np.zeros((counts.shape[0],counts.shape[1],width_after))
    for i in range(width):
        if i//dt<width_after:
            counts_after[:,:,i//dt]+=counts[:,:,i]
        
    return counts_after



        
        
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
    
    if np.min(np.sum(counts,axis=1))==0:
        print('Counts = 0 for some t, i.')
    
    
    return counts 


# def counts_bootstrap_lineage(counts, Nlins):
      
#     ND,slmax,tmax  = counts.shape
      
#     totcounts = np.sum(counts,axis=1)
#     aux= np.random.choice(slmax, slmax) # Bootstrapping: Sample lineages with replacement 
#     set_lins=np.array_split(aux ,Nlins)
#     counts_superlin=[]
#     for s in set_lins:
#         counts_superlin.append(np.sum(counts[:,s,:],axis=1))
#     counts_superlin=np.array(counts_superlin) #Nlins, ND, tmax
    
#     counts_superlin+=1
#     for i in range(ND):
#         for t in range(tmax):
#             counts_superlin[:,i,t]*=totcounts[i,t]/np.sum(counts_superlin[:,i,t])
            
#     counts_superlin = counts_superlin.transpose([1,0,2])
    
#     return counts_superlin





# def construct_superfreq(counts, Nlins):
      
#     ND,slmax,tmax  = counts.shape
      
#     aux= np.random.choice(slmax, slmax) # Bootstrapping: Sample lineages with replacement 
#     set_lins=np.array_split(aux ,Nlins)
#     counts_superlin=[]
#     for s in set_lins:
#         counts_superlin.append(np.sum(counts[:,s,:],axis=1))
#     counts_superlin=np.array(counts_superlin)
    
    
#     # Pseudo count
#     counts_deme= np.sum(counts_superlin,axis=0)
#     for l in range(Nlins):
#         for i in range(ND):
#             for t in range(tmax):
#                 if counts_superlin[l,i,t]==0:
#                     counts_superlin[l,i,t]+=1
#                 #counts_superlin[l,i,t]+=1
    
#     B = np.zeros((tmax,Nlins,ND))
#     for t in range(tmax):
#         for i in range(ND):
#             if counts_deme[i,t]>0:
#                 B[t,:,i]= counts_superlin[:,i,t]/counts_deme[i,t] 
#             else:
#                 B[t,:,i]=np.nan
                
#     return B


# def construct_superfreq_nonpseudo(counts, Nlins):
      
#     ND,slmax,tmax  = counts.shape
      
#     aux= np.random.choice(slmax, slmax) # Bootstrapping: Sample lineages with replacement 
#     set_lins=np.array_split(aux ,Nlins)
#     counts_superlin=[]
#     for s in set_lins:
#         counts_superlin.append(np.sum(counts[:,s,:],axis=1))
#     counts_superlin=np.array(counts_superlin)
    
#     counts_deme= np.sum(counts_superlin,axis=0)
    
#     B = np.zeros((tmax,Nlins,ND))
#     for t in range(tmax):
#         for i in range(ND):
#             if counts_deme[i,t]>0:
#                 B[t,:,i]= (counts_superlin[:,i,t])/counts_deme[i,t] 
#             else:
#                 B[t,:,i]=np.nan
                
#     return B



def infer_EM_LS(counts,EM_kws, Qplot='n'):
    
    itermax=EM_kws['itermax']
    Nlins=EM_kws['Nlins']
    filename = EM_kws['filename']
    outpath= EM_kws['outpath']
    noisemode = EM_kws['noisemode']
    Path(outpath).mkdir(parents=True, exist_ok=True)
            
    ND,slmax,tmax  = counts.shape
    print("ND,slmax,tmax ",ND,slmax,tmax )
    
    B = construct_superfreq_nonpseudo(counts, Nlins)
    aux=B.flatten()
    print("{} % of components of non-pseudo B  are zero.".format(round(100* len(aux[aux==0.0])/len(aux))))

    if Qplot=='y':
        for i in range(Nlins):
            plt.plot(B[:,i,0])
        plt.show()
    
    
    ##APPLY HMM 
    
    res_A_EM=[]
    res_Ne_EM=[]
    res_A_LS=[]
    res_A_LSWF=[]
    res_Ne_LSWF =[]
    
    counts_deme= np.sum(counts,axis=1)
    counts_deme+=1
    
    terminal_com=''
    for iter in range(itermax):
        setting = filename+'_iter'+str(iter)
        
        B = construct_superfreq(counts, Nlins)
            
        A_LS=lindyn_qp(B, lam=0)
        
        A_LSWF, Ne_LSWF=LSWF(B)
        
        
        lnLH_record, A_EM, Ne_EM=Kalman_EM(B,counts_deme,em_step_max=30, terminate_th=0.0001,noisemode=noisemode,iterate=10)
        res_A_EM.append(A_EM)
        res_Ne_EM.append(Ne_EM)
        res_A_LS.append(A_LS)
        res_A_LSWF.append(A_LSWF)
        res_Ne_LSWF.append(Ne_LSWF)
        
        
    res_A_EM = np.array(res_A_EM)   
    res_Ne_EM = np.array(res_Ne_EM)
    res_A_LS = np.array(res_A_LS)  
    res_A_LSWF = np.array(res_A_LSWF) 
    res_Ne_LSWF = np.array(res_Ne_LSWF)
    
    
    np.save(outpath+ 'A_EM_'+filename+'.npy', res_A_EM)  
    np.save(outpath+'Ne_EM_'+filename+'.npy', res_Ne_EM )  
    np.save(outpath+'A_LS_'+filename+'.npy', res_A_LS) 
    np.save(outpath+'A_LSWF_'+filename+'.npy', res_A_LSWF) 
    np.save(outpath+'Ne_LSWF_'+filename+'.npy', res_Ne_LSWF) 
    np.save(outpath+ 'countsdeme_'+filename+'.npy', counts_deme)
    
    now = datetime.now()
    with open(outpath+'log_'+filename+'.txt', 'w') as f:
        f.write('filename, '+filename)
        f.write('\n')
        f.write('Data is created on '+now.strftime("%m/%d/%Y, %H:%M:%S"))
        f.write('\n')
        f.write('itermax = '+str(itermax))
        f.write('\n')
        f.write('ND,slmax,tmax'+str(ND)+','+str(slmax)+','+str(tmax))
        f.write('\n')
        f.write('Nlins= '+str(Nlins))
        f.write('\n')
        f.write('noisemode= '+str(noisemode))
        
        
def load_MCMC(loadMCMC_kws,showlogLH=False):
    
    dir_IO =  loadMCMC_kws['dir_IO']
    burn_in = loadMCMC_kws['burn_in']
    modename = loadMCMC_kws['modename']
    itermax = loadMCMC_kws['itermax']
    
    dir_MCMC='HMM_MCMC/'
        
    res_logLH_mcmc=[]
    for iter in range(itermax):

        setting = modename+'_iter'+str(iter)
        if iter==itermax-1:
            print('File lastly loaded = ', setting)
      
        A_mcmc=np.loadtxt(dir_MCMC+'output/'+dir_IO+'/A_'+setting+'.csv',delimiter=',')
        Ne_mcmc=np.loadtxt(dir_MCMC+'output/'+dir_IO+'/Ne_'+setting+'.csv',delimiter=',')
        
        C_mcmc=np.loadtxt(dir_MCMC+'output/'+dir_IO+'/C_'+setting+'.csv',delimiter=',')
        
        logLH_mcmc=np.loadtxt(dir_MCMC+'output/'+dir_IO+'/logLH_'+setting+'.csv',delimiter=',')
        
        ND = len(Ne_mcmc[0])
       
        mcmcsteps=len(A_mcmc)
   
        if iter==0:
            res_A_mcmc=np.copy(A_mcmc[int(burn_in*mcmcsteps):,:])
            res_Ne_mcmc=np.copy(Ne_mcmc[int(burn_in*mcmcsteps):,:])
            res_logLH_mcmc.append(logLH_mcmc)
        else:
            res_A_mcmc=np.concatenate((res_A_mcmc,A_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)
            res_Ne_mcmc=np.concatenate((res_Ne_mcmc,Ne_mcmc[int(burn_in*mcmcsteps):,:]),axis=0)
            res_logLH_mcmc.append(logLH_mcmc)
    res_logLH_mcmc=np.array(res_logLH_mcmc)
    

    res_A_mcmc = np.array([ i.reshape((ND,ND)) for i in res_A_mcmc])
    
    if showlogLH==True:
        plt.figure(figsize=(6,3))
        x_BI = round(burn_in*len(logLH_mcmc))
        for idx, logLH_mcmc in enumerate(res_logLH_mcmc):
            if idx<10:
                plt.plot(logLH_mcmc[:x_BI,0],logLH_mcmc[:x_BI,1],alpha=0.2)
                plt.plot(logLH_mcmc[x_BI:,0],logLH_mcmc[x_BI:,1])
        plt.vlines(x=logLH_mcmc[round(burn_in*len(logLH_mcmc)),0], 
                   ymin = np.min(res_logLH_mcmc[:,:,1]),ymax =np.max(res_logLH_mcmc[:,:,1])
                  ,label='Burn-in',color='k')
        plt.ylabel('LogLH')
        plt.xlabel('MCMC steps')
        plt.legend()
        plt.show()

    return res_A_mcmc, res_Ne_mcmc,C_mcmc






########################
#functions that prepare MCMC input.

def create_bs_counts(counts,itermax=25):
    bs_counts=[]
    aux = np.array(range(counts.shape[1]))
    for iter in range(itermax):
        bs_counts.append(counts[:,np.random.choice(aux,size=counts.shape[1],replace =True),:].copy())
    return bs_counts





# MUTATION HOKUSAI
def prepare_MCMC(res_counts,totcounts, mcmc_kws,df_sampled=None,Qplot='n'):
   
    
    MCMC_dir = 'HMM_MCMC/'
    
    dir_IO=mcmc_kws['dir_IO']
    QDBlist=mcmc_kws['QDBlist']
    noisemodelist =mcmc_kws['noisemodelist']
    mcmcsteps=mcmc_kws['mcmcsteps']
    Q_hokusai_local=mcmc_kws['Q_hokusai_local']
    
    print(Q_hokusai_local)
        
    itermax=len(res_counts)
    
    inpath=MCMC_dir+'input/'+dir_IO+'/'
    Path(inpath).mkdir(parents=True, exist_ok=True)
    
    ND,Nmut,tmax  = res_counts[0].shape
    
    list_infilename =[]
    
    if df_sampled is not None:
        df_sampled.to_csv(inpath+'mutations_used.csv')
    
    for iter, counts in enumerate(res_counts):
        
        counts  = counts.transpose()
        shapelst=list(counts.shape)
        
        aux=counts[:,0,:].copy()
        for i in range(1,Nmut):
            aux=np.concatenate((aux,counts[:,i,:]),axis=0)
            
        
        
        infilename = 'iter'+str(iter)
        list_infilename.append(infilename)
        np.savetxt(inpath+'counts_'+infilename+'.csv', aux, fmt="%d", delimiter=",")
        np.savetxt(inpath+'totcounts_'+infilename+'.csv', totcounts.transpose(), fmt="%d", delimiter=",")
        np.savetxt(inpath+'shape_'+infilename+'.csv', shapelst, fmt="%d", delimiter=",")
        
        
    #sh files 
    terminal_command=''
    QDBlist=mcmc_kws['QDBlist']
    noisemodelist =mcmc_kws['noisemodelist']
    mcmcsteps=mcmc_kws['mcmcsteps']
    Path(MCMC_dir+'sh/'+dir_IO).mkdir(parents=True, exist_ok=True)
    for iter in range(itermax):
        for noisemode in noisemodelist:
            for QDB in QDBlist:
                
                sh_filename=dir_IO+'/noisemode{}_{}_iter{}.sh'.format(noisemode,QDB,iter)
              
                text_file = open(MCMC_dir+'sh/'
                                 +sh_filename, "w")
                
                outfilename = 'noisemode{}_{}_'.format(noisemode,QDB)+list_infilename[iter]
                mcmc_option = ' -f {} -g {} -d {} -m {} -n {} -D {}'.format(list_infilename[iter], 
                                                                            outfilename,
                                                                      dir_IO, 
                                                                      mcmcsteps,
                                                                      noisemode ,
                                                                      QDB)
                
                if Q_hokusai_local=='local':
                    contents_sh ='./a.out'+mcmc_option+';'
                elif Q_hokusai_local=='hokusai':
                    contents_sh = '#!/bin/sh\n'
#                     contents_sh +='#------ pjsub option --------#\n'
#                     contents_sh +='#PJM -L rscunit=bwmpc\n'
#                     contents_sh +='#PJM -L elapse=24:00:00\n'
#                     contents_sh +='#PJM -L vnode=1\n'
#                     contents_sh +='#PJM -L vnode-core=2\n'
  
#                     contents_sh +='#------- Program execution ---#\n'
                    contents_sh +='#------ pjsub option --------#\n'
                    contents_sh +='#SBATCH --partition=mpc\n'
                    contents_sh +='#SBATCH --account RB230027\n'
                    contents_sh +='#SBATCH --time=24:00:00\n'
                    contents_sh +='#SBATCH --nodes=1\n'
  
                    contents_sh +='#------- Program execution ---#\n'
                    contents_sh +='module load intel\n'
                    contents_sh +='./a.out'+mcmc_option+';'
                    
                #write string to file
                text_file.write(contents_sh)
                #close file
                text_file.close()

                if Q_hokusai_local=='local':
                    terminal_command+='sh sh/'+sh_filename+';'
                elif Q_hokusai_local=='hokusai':
                    terminal_command+='sbatch sh/'+sh_filename+';'
                    
    return terminal_command










def prepare_EM(res_counts_BS, totcounts, EM_kws):
    dir_IO = EM_kws['dir_IO']
    noisemodelist = EM_kws['noisemodelist']
    num_run= EM_kws['num_run']
    filename = EM_kws['filename']
    Q_hokusai_local=EM_kws['Q_hokusai_local']
    
    
    if 'ridgelist' in EM_kws.keys():
        ridgelist=EM_kws['ridgelist']
    else:
        ridgelist=[0]
        
    if 'ridge_mat' in EM_kws.keys():
        ridge_mat=EM_kws['ridge_mat']
    else:
        ridge_mat=np.zeros((1,1,1))
        
    if 'regionnumlist' in EM_kws.keys():
        regionnumlist=EM_kws['regionnumlist']
        
    else:
        regionnumlist=np.array([])
        
    if 'scriptname' in EM_kws.keys():
        EMscriptname=EM_kws['scriptname']
    else:
        EMscriptname='EM.py'
        
    if 'penalty_mode' in EM_kws.keys():
        penalty_mode=EM_kws['penalty_mode']
    else:
        penalty_mode='L2'
        
    

        
    print(Q_hokusai_local)
    
    HMM_EM_dir ='HMM_EM/input/'
    HMM_EM_sh_dir ='HMM_EM/sh/'
    Path(HMM_EM_dir+dir_IO).mkdir(parents=True, exist_ok=True)
    Path(HMM_EM_sh_dir+dir_IO).mkdir(parents=True, exist_ok=True)
    
    res_counts_BS=np.array(res_counts_BS)
    split_iter = np.array_split(range(len(res_counts_BS)), num_run)
    
    terminal_command=''

    if 'testdata' in EM_kws.keys() and EM_kws['testdata'] !=np.array([]):
        #HMM_EM_dir_test ='HMM_EM/test/'
        #Path(HMM_EM_dir +dir_IO).mkdir(parents=True, exist_ok=True)
        res_counts_BS_test=np.array(EM_kws['testdata'])
        for run, sp  in enumerate(split_iter):
            np.save(HMM_EM_dir +dir_IO+'/'+'testcounts_'+filename+str(run)+'.npy',res_counts_BS_test[sp])
            #np.save(HMM_EM_dir_test +dir_IO+'/'+'testcounts_'+filename+str(run)+'.npy',res_counts_BS_test[sp])
    
    for run, sp  in enumerate(split_iter):
        
        np.save(HMM_EM_dir+dir_IO+'/'+'counts_'+filename+str(run)+'.npy',res_counts_BS[sp])
        np.save(HMM_EM_dir+dir_IO+'/'+'totcounts_'+filename+str(run)+'.npy',totcounts)
        np.save(HMM_EM_dir+dir_IO+'/'+'regionnumlist_'+filename+str(run)+'.npy',np.array(regionnumlist))
        np.save(HMM_EM_dir+dir_IO+'/'+'ridgemat_'+filename+str(run)+'.npy',ridge_mat)

        for ridge in ridgelist:
            for noisemode in noisemodelist:
                setting = ' '+dir_IO+'/'+' run{} {} {}'.format(run, noisemode, ridge)
                if penalty_mode=='L1':
                    setting=setting +' '+penalty_mode
                #print(EMscriptname)
                    

                if Q_hokusai_local=='local':
                    aux = "python "+EMscriptname+setting
                elif Q_hokusai_local=='hokusai':

                    aux = "#!/bin/sh\n#------ pjsub option --------#\n#SBATCH --partition=mpc\n"+"#SBATCH --account RB230027\n"+"#SBATCH --time=24:00:00\n"+"#SBATCH --time=24:00:00\n"+"#SBATCH --nodes=1\n"+"#------- Program execution -------#\n"+"module load intel\n"+"export PYENV_ROOT=\"$HOME/.pyenv\"\nexport PATH=\"$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH\"\neval \"$(pyenv init -)\"\nexport PATH=\"$PYENV_ROOT/versions/miniconda3-4.7.12/bin/:$PYENV_ROOT/versions/miniconda3-4.7.12/condabin:$PATH\"\nexport PATH=\"$PYENV_ROOT/versions/3.9.12/bin/:$PATH\"\n\nexport OMP_NUM_THREADS=1\npython "+EMscriptname+setting
                    
                    # aux = "#!/bin/sh\n#------ pjsub option --------#\n#PJM -L rscunit=bwmpc\n#PJM -L rscgrp=batch\n#PJM -L vnode=1\n#PJM -L vnode-core=5\n#PJM -L vnode-mem=4400Mi\n#PJM -L elapse=20:00:00\n#PJM -g Q23444\n#PJM -j\n#------- Program execution -------#\nexport PYENV_ROOT=\"$HOME/.pyenv\"\nexport PATH=\"$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH\"\neval \"$(pyenv init -)\"\nexport PATH=\"$PYENV_ROOT/versions/miniconda3-4.7.12/bin/:$PYENV_ROOT/versions/miniconda3-4.7.12/condabin:$PATH\"\nexport PATH=\"$PYENV_ROOT/versions/3.9.12/bin/:$PATH\"\n\nexport OMP_NUM_THREADS=1\npython "+EMscriptname+setting                    

                shfilename =HMM_EM_sh_dir+dir_IO+'/'+'noisemode{}'.format(noisemode)+'_ridge{}'.format(ridge)+'_'+filename+str(run)
                text_file=open(shfilename+".sh", "w")
                text_file.write(aux)
                text_file.close()


                if Q_hokusai_local=='local':
                    terminal_command=terminal_command+'sh '+shfilename+".sh;"
                elif Q_hokusai_local=='hokusai':
                    terminal_command=terminal_command+'sbatch '+shfilename+".sh;"
                
    return terminal_command



###########################
from scipy.stats import nbinom
import scipy
def calc_logLH(r1,r2,R1,R2, s, k, meanfit,Dt=1):
    mu = r1*R2/R1 *np.exp((s-meanfit)*Dt)
    var = k*mu

    p = mu/var
    n = mu*mu/(var-mu)
    
    return scipy.stats.nbinom.logpmf(r2, n=n, p=p)

def fitness_estimator(counts_allx ,totcounts_allx,smax=.25):

    freq_allx = counts_allx@np.diag(1/totcounts_allx)

    Dt=1
    meanfit=[]
    kappa=[]
    for t in range(1,freq_allx.shape[1]):
        nonzero = list(set(np.where(freq_allx[:,t]>0)[0]).intersection(set(np.where(freq_allx[:,t-1]>0)[0])))

        f1 = freq_allx[nonzero,t-1].copy()
        f2 = freq_allx[nonzero,t].copy()
        meanfit.append( (np.median(np.log(f1) - np.log(f2)))/Dt )

        dphi = np.sqrt(freq_allx[nonzero,t])-np.sqrt(freq_allx[nonzero,t-1])
        med_dphi  = np.median(dphi)
        kappa.append( np.max([ 4*totcounts_allx[t]*2.1981*np.median((dphi  - med_dphi)**2), 1.01]) ) 


    slist=list(np.round(list(np.linspace(-smax,smax,301)),3))


    logLH=np.zeros((freq_allx.shape[0],len(slist)))
    for l in range(freq_allx.shape[0]):
        for si, s in enumerate(slist):
            aux=0
            for t in range(1,freq_allx.shape[1]):
                aux += calc_logLH(r1=counts_allx[l,t-1],r2=counts_allx[l,t],R1=totcounts_allx[t-1],R2=totcounts_allx[t], s=s, k=kappa[t-1], meanfit=meanfit[t-1],Dt=1)
            logLH[l,si] = aux
            
    # For numerical stability, max Log LH is set to 0, and then the probability is computed
    logLH_norm=logLH.copy() 
    for l in range(freq_allx.shape[0]):
        logLH_norm[l] -= np.max(logLH_norm[l])
    pr_norm =  np.exp(logLH_norm)
    for l in range(freq_allx.shape[0]):
        pr_norm[l]*=1./np.sum(pr_norm[l])

    
    pval=[]
    s_est=[]
    for l in range(freq_allx.shape[0]):
        logLH_s0=logLH_norm[l,slist.index(0.0)]
        LLratio=logLH_s0-logLH_norm[l,:]
        pval.append(np.sum(pr_norm[l,np.where(LLratio>0)[0]])) # pval is given by Prob(s with LH(s=0) > LH(s)) 
        s_est.append(slist[list(logLH_norm[l,:]).index(np.max(logLH_norm[l,:]))])
        
    df=pd.DataFrame()
    df['s']=s_est.copy()
    df['pval']=pval.copy()
    return df




def fitness_estimator_plot(counts_allx,totcounts_allx,smax,pval,outpath,filename):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)

    df = fitness_estimator(counts_allx,totcounts_allx ,smax)

    fig, axs = plt.subplots(ncols=3, figsize=(8, 2.5))

    smax= np.max(df['s'])
    

    freq=counts_allx@np.diag(1/totcounts_allx)
    # Normalize selection coefficients
    norm = mcolors.Normalize(vmin=-smax, vmax=smax)
    cmap = plt.colormaps['coolwarm'] 
    # Plot frequency data with colors mapped to selection coefficient
    for i in range(freq.T.shape[1]):
        color = cmap(norm(df['s'].iloc[i]))
        if df['pval'].iloc[i] > pval:
            axs[0].plot(freq[ i], color=color)
        else:
            axs[1].plot(freq[ i], color=color)
    axs[0].set_xlabel('Week')
    axs[1].set_xlabel('Week')

    axs[0].set_title('Non-significant, '+'p > ' + str(pval))
    axs[1].set_title('Significant, '+'p < ' + str(pval))

    axs[0].set_ylim(0,)
    axs[1].set_ylim(0,)
    axs[0].set_ylabel('Frequency')

    above_pval = df[df['pval'] > pval]
    below_pval = df[df['pval'] < pval]
    sc1 = axs[2].scatter(above_pval['s'], above_pval['pval'], label='p > ' + str(pval), marker='o', facecolors='none', edgecolors=cmap(norm(above_pval['s'])),s=50)
    sc2 = axs[2].scatter(below_pval['s'], below_pval['pval'], label='p < ' + str(pval), marker='x', c=cmap(norm(below_pval['s'])),s=50)

    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_xlim(-smax-0.05, smax+0.05)
    axs[2].set_xlabel('Selection Coefficient')
    axs[2].set_ylabel('p value')

    # Manually create a colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[2])
    cbar.set_label('Selection Coefficient')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', label='p > ' + str(pval)),
                       plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='none', markeredgecolor='black', label='p < ' + str(pval))]
    axs[2].legend(handles=legend_elements,bbox_to_anchor=[0.5,1.2],ncol=2,loc='upper center',columnspacing=.4,handletextpad=0.0,fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath+filename+'/'+'filter'+filename+'.pdf')
    plt.show()
    
    return df


def calc_ALS_BS(counts, totcounts,BS='n'):
    
    if BS=='y':
        Nlins=counts.shape[1]
        bs=np.random.choice(range(Nlins), size=Nlins,replace=True)
        counts=counts[:,bs,:]
        
    B = (counts.copy()).transpose([2,1,0])
    T, Nlins, ND=B.shape

    
    for i in range(ND):
        for t in range(T):
            if totcounts[i,t]>0:
                B[t,:,i]*=1.0/totcounts[i,t]

    A_LS=lindyn_qp(B,lam=0)

    return A_LS