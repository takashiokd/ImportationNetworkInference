
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
#from modules.LDS import lindyn_qp, Kalman_EM, update_A
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl

def CI(data,alpha):
    sortdata=np.sort(data)
    return [sortdata[round(0.5*alpha*len(data))],sortdata[-round(0.5*alpha*len(data))]]

from scipy.linalg import null_space


from shapely.geometry import Polygon
from matplotlib.patheffects import withStroke


######################
# # selection

def updatedensity_ode(xini,t0,t1,A,k, sigma):
    ND = len(A)


    
    dt =0.05
    res=[]
    x =np.copy(xini)
    res.append(np.copy(x))
    titermax =int( (t1-t0-1)/dt)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        x_aux = np.copy(x)
        for i in range(ND):
            x[i] += k*x_aux[i]*dt
            for j in range(ND):
                if j!=i:
                    x[i] += (1+sigma)*A[i,j]*x_aux[j]*dt
        if int(t)==t:          
            res.append(np.copy(x))
    return np.array(res)
  
# def updatefreq_ode(freqini,t0,t1,A,k, sigma):
#     ND = len(A)

        
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



             
def sim_freq_ode_old(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05, freq_control=[]):
    ND = len(A)

    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt*np.heaviside(freq_aux[i]-fcr[i], 0)
            for j in range(ND):
                freq[i]  += k*A[i,j]*freq_aux[j]*dt
                
            if len(freq_control)>0:
                if freqini[i]>0:
                    freq[i]=freq_control[int(t),i]

        if int(t)==t:          
            res.append(np.copy(freq))


    return np.array(res) 


def sim_freq_ode_new(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05):
    ND = len(A)
 
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt*np.heaviside(freq_aux[i]-fcr[i], 0)
            for j in range(ND):
                if i!=j:
                    freq[i]+= A[i,j]*(freq_aux[j]-freq_aux[i])*dt + (k-1)*A[i,j]*freq_aux[j]*(1-freq_aux[i])*dt
        if int(t)==t:          
            res.append(np.copy(freq))
    return np.array(res) 


             
def sim_freq_linearode_old(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05):
    ND = len(A)

    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*dt
            for j in range(ND):
                 freq[i]  += k*A[i,j]*freq_aux[j]*dt
        if int(t)==t:          
            res.append(np.copy(freq))
    return np.array(res) 

             
def sim_freq_linearode_new(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05):
    ND = len(A)

    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*dt
            for j in range(ND):
                 if i!=j:
                    freq[i]  += A[i,j]*(freq_aux[j]-freq_aux[i])*dt + (k-1)*A[i,j]*freq_aux[j]*dt
        if int(t)==t:          
            res.append(np.copy(freq))
    return np.array(res) 


def sim_freq_approxode_old(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05):
    Atemp = np.copy(A)
    
    ND = len(A)

        
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    
    
    Adiag=0
    for i in range(ND):
        Adiag+=A[i,i]/ND
    for i in range(ND):        
        A[i,i] = Adiag
        
    
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
            freq[i] += (s+A[i,i])*freq_aux[i]*dt
            if freqini[i]==0:
                for j in range(ND):
                        if freqini[j]>0:
                            freq[i]  += k*A[i,j]*freq_aux[j]*dt
        if int(t)==t:          
            res.append(np.copy(freq))
            
            
        
    for i in range(ND): 
        for j in range(ND):
            A[i,j] = Atemp[i,j]
    
    return np.array(res) 

# def sim_freq_approxode_new(freqini,t0,t1,A,k, s,fcr=[0]*9, dt =0.05):

#     ND = len(A)

        
#     res=[]
#     freq =np.copy(freqini)
#     res.append(np.copy(freq))
#     titermax =int( (t1-t0-1)/dt)+1
#     ND = len(A)
    

    
#     for titer in range(titermax):
#         t = t0+dt*(titer+1)
#         freq_aux = np.copy(freq)
#         for i in range(ND):
#             freq[i] += s*freq_aux[i]*dt
#             if freqini[i]==0:
#                 for j in range(ND):
#                         if freqini[j]>0:
#                             freq[i]  += k*A[i,j]*freq_aux[j]*dt
                            
#         if int(t)==t:          
#             res.append(np.copy(freq))

#     return np.array(res) 


def sim_freq_stochastic_old(freqini,t0,t1,A,k, s,Ne=[100]*9, dt =0.05):
    ND = len(A)

    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt + np.sqrt(freq_aux[i]*(1-freq_aux[i])*dt/Ne[i])*np.random.normal()
            for j in range(ND):
                freq[i]  += k*A[i,j]*freq_aux[j]*dt
                    
            if freq[i]<0:
                freq[i]=0
            if freq[i]>1:
                freq[i]=1
                
        if int(t)==t:          
            res.append(np.copy(freq))
            
            
    return np.array(res)  



def sim_freq_stochastic_new(freqini,t0,t1,A,k, s,Ne=[100]*9, dt =0.05):
    ND = len(A)

        
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt + np.sqrt(freq_aux[i]*(1-freq_aux[i])*dt/Ne[i])*np.random.normal()
            for j in range(ND):
                if j!=i:
                    freq[i]  += A[i,j]*(freq_aux[j]-freq_aux[i])*dt + (k-1)*A[i,j]*freq_aux[j]*(1-freq_aux[i])*dt
                    
            if freq[i]<0:
                freq[i]=0
            if freq[i]>1:
                freq[i]=1
                
        if int(t)==t:          
            res.append(np.copy(freq))
            
            
    return np.array(res)  



def sim_freq_stochastic_migred_old(freqini,t0,t1,A,k, s,Ne=[100]*9, dt =0.05, t_reduce=3,frac_reduce=0.1):
    ND = len(A)

        
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    
    
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        
        if t<t_reduce:
            red=1
        else:
            red = frac_reduce
        
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt + np.sqrt(freq_aux[i]*(1-freq_aux[i])*dt/Ne[i])*np.random.normal()
            for j in range(ND):
                    freq[i]  += red*k*A[i,j]*freq_aux[j]*dt
                    
            if freq[i]<0:
                freq[i]=0
            if freq[i]>1:
                freq[i]=1
                
        if int(t)==t:          
            res.append(np.copy(freq))
            
    return np.array(res)  




def sim_freq_stochastic_migred_new(freqini,t0,t1,A,k, s,Ne=[100]*9, dt =0.05, t_reduce=3,frac_reduce=0.1):
    ND = len(A)

        
    res=[]
    freq =np.copy(freqini)
    res.append(np.copy(freq))
    titermax =int( (t1-t0-1)/dt)+1
    ND = len(A)
    
    
    for titer in range(titermax):
        t = t0+dt*(titer+1)
        freq_aux = np.copy(freq)
        
        if t<t_reduce:
            red=1
        else:
            red = frac_reduce
        
        for i in range(ND):
    
            freq[i] += s*freq_aux[i]*(1-freq_aux[i])*dt + np.sqrt(freq_aux[i]*(1-freq_aux[i])*dt/Ne[i])*np.random.normal()
            for j in range(ND):
                if j!=i:
                    freq[i]  += A[i,j]*(freq_aux[j]-freq_aux[i])*dt + (k-1)*A[i,j]*freq_aux[j]*(1-freq_aux[i])*dt
                    
            if freq[i]<0:
                freq[i]=0
            if freq[i]>1:
                freq[i]=1
                
        if int(t)==t:          
            res.append(np.copy(freq))
            
    return np.array(res)  




def fit_sweep( mode, focal_variant,index, x_actual, res_A, t0,t1, tfit0,tfit1, round_max=10,filename='demo',k_fixed='n',sigma_fixed ='n',outpath='fig/selection/'):
    Path(outpath).mkdir(parents=True, exist_ok=True)
    
    ND = len(res_A[0])
    itermax = len(res_A)
    
    err_min=100000000

    nonzeromin =[]
    aux = x_actual.flatten()
    for i in aux:
        if i>0:
            nonzeromin.append(i)
    nonzeromin =np.min(nonzeromin)
            
    ini_old = np.copy(x_actual[tfit0])
    
    
    for idx, i in enumerate(ini_old):
        if i<nonzeromin:
            ini_old[idx] =nonzeromin
            
    k_old =0.8
    sigma_old = -0.5
    k_res=[]
    sigma_res=[]
    err_res=[]
    
    for rd in range(round_max):
        
        if k_fixed =='n':
            k= k_old*np.exp(np.random.normal(0,0.05))
        else:
            k =k_fixed
        
        ini = ini_old*[np.exp(np.random.normal(0,0.1)) for i in range(ND)]
      
        if sigma_fixed =='n':
            sigma = sigma_old + np.random.normal(0,0.05)
            if sigma<-1:
                sigma=-1
        else:
            sigma = sigma_fixed
            
        err=0

        for iter in range(itermax):
            A = np.copy(res_A[iter])
            
            if mode=='density':
                x_predicted=updatedensity_ode(ini,tfit0,tfit1,A,k, sigma)
                err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2)/x_actual[tfit0:tfit1].flatten())
            elif mode=='frequency':
                x_predicted = updatefreq_ode(ini,tfit0,tfit1,A,k, sigma)
                err_mag  =  np.array([i*(1-i) for i in x_actual[tfit0:tfit1].flatten() ])
                err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2)/err_mag )
                
          #  err+= np.sum(np.power(x_actual[tfit0:tfit1].flatten() -x_predicted[:].flatten(),2))
            
            

        if err_min>err:
            err_min=err
            ini_old = np.copy(ini)
            k_old = k
            sigma_old = sigma

        k_res.append(k_old)
        err_res.append(err_min)
        sigma_res.append(sigma_old)
        
    plt.figure(figsize = (16,3))
    plt.subplot(1, 3, 1)
    plt.plot(err_res)
    plt.ylabel('err')
    
    plt.subplot(1, 3, 2)
    plt.plot(sigma_res)
    plt.ylabel('sigma')
    
    plt.subplot(1, 3, 3)
    plt.plot(k_res)
    plt.ylabel('k')
    plt.show()
        
    ini_opt = np.copy(ini_old)
    k_opt = np.round(k_old,2)
    sigma_opt = np.round(sigma_old,2)
    
    
    t_predict=t1-(t0+tfit0)
    trajs_opt=[]
    for iter in range(itermax):
        A = np.copy(res_A[iter])
        if mode=='density':
            x_predicted=updatedensity_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
        elif mode=='frequency':
            x_predicted=updatefreq_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
        trajs_opt.append(x_predicted)
    trajs_opt = np.copy(np.array(trajs_opt))

    
    for yscale_mode in ['linear','log']:
    
        for i in range(ND):
            plt.scatter(range(t0,t1),x_actual[:,i],color=CB_color_cycle[i],label=index[i])
            plt.plot(range(t0,t1),x_actual[:,i],'-',color=CB_color_cycle[i],alpha=0.2)
            plt.plot(range(t0+tfit0,t0+tfit1),np.mean(trajs_opt,axis=0)[:tfit1-tfit0,i],'--',color=CB_color_cycle[i])
            #plt.plot(range(t0+tfit1-1,t0+tfit0+t_predict),np.mean(trajs_opt,axis=0)[tfit1-tfit0-1:,i],'--',color=CB_color_cycle[i],alpha=0.3)
        plt.ylim(np.min(x_actual),np.max(x_actual)*1.2)

        plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='fit')
        #plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='prediction')
        plt.title('k='+str(k_opt)+', sigma='+str(sigma_opt)+'\n '+str(tfit1-tfit0)+' timepoints are used in fitting')
        plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0))
        
        if yscale_mode=='log':
            plt.yscale('log')
            if mode=='density':
                plt.ylabel('Density of '+focal_variant)
                plt.savefig(outpath+'log_densityspace'+filename+'.pdf',bbox_inches='tight')
            elif mode=='frequency':
                plt.ylabel('Frequency of '+focal_variant)
                plt.savefig(outpath+'log_freqspace'+filename+'.pdf',bbox_inches='tight')
            plt.show()
        else:
            if mode=='density':
                plt.ylabel('Density of '+focal_variant)
                plt.savefig(outpath+'densityspace'+filename+'.pdf',bbox_inches='tight')
            elif mode=='frequency':
                plt.ylabel('Frequency of '+focal_variant)
                plt.savefig(outpath+'freqspace'+filename+'.pdf',bbox_inches='tight')
            plt.show()
            
               
#     for i in range(ND):
#         plt.scatter(range(t0,t1),x_actual[:,i],color=CB_color_cycle[i],label=index[i])
#         plt.plot(range(t0+tfit0,t0+tfit1),np.mean(trajs_opt,axis=0)[:tfit1-tfit0,i],color=CB_color_cycle[i],alpha=0.3)
#         plt.plot(range(t0+tfit1-1,t0+tfit0+t_predict),np.mean(trajs_opt,axis=0)[tfit1-tfit0-1:,i],'--',color=CB_color_cycle[i],alpha=0.3)
#     plt.ylim(np.min(x_actual),np.max(x_actual)*1.2)
    
#     plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'-',color='black',label='fit')
#     plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='prediction')
#     plt.title('k='+str(k_opt)+', sigma='+str(sigma_opt)+'\n '+str(tfit1-tfit0)+' timepoints are used in fitting')
#     plt.legend()
#     if mode=='density':
#         plt.ylabel('Density of '+focal_variant)
#         plt.savefig('fig/selection/densityspace'+filename+'.pdf',bbox_inches='tight')
#     elif mode=='frequency':
#         plt.ylabel('Frequency of '+focal_variant)
#         plt.savefig('fig/selection/freqspace'+filename+'.pdf',bbox_inches='tight')
#     plt.show()
    
    
# def fit_sweep_density(focal_variant,index, i_actual, res_A, t0,t1, tfit0,tfit1, round_max=10,filename='demo',k_fixed='n',sigma_fixed ='n'):
#     Path('fig/selection/').mkdir(parents=True, exist_ok=True)
    
#     ND = len(res_A[0])
#     itermax = len(res_A)
    
#     err_min=100000000

#     nonzeromin =[]
#     aux = i_actual.flatten()
#     for i in aux:
#         if i>0:
#             nonzeromin.append(i)
#     nonzeromin =np.min(nonzeromin)
            
#     ini_old = np.copy(i_actual[tfit0])
    
    
#     for idx, i in enumerate(ini_old):
#         if i<nonzeromin:
#             ini_old[idx] =nonzeromin
            
#     k_old =1.0
#     sigma_old = 0
#     k_res=[]
#     sigma_res=[]
#     err_res=[]
    
#     for rd in range(round_max):
        
#         if k_fixed =='n':
#             k= k_old*np.exp(np.random.normal(0,0.05))
#         else:
#             k =k_fixed
        
#         ini = ini_old*[np.exp(np.random.normal(0,0.05)) for i in range(ND)]
      
#         if sigma_fixed =='n':
#             sigma = sigma_old + np.random.normal(0,0.1)
#             if sigma<-1:
#                 sigma=-1
#         else:
#             sigma = sigma_fixed
            
#         err=0

#         for iter in range(itermax):
#             A = np.copy(res_A[iter])
#             i_predicted=updatedensity_ode(ini,tfit0,tfit1,A,k, sigma)
#             err+= np.sum(np.power(i_actual[tfit0:tfit1].flatten() - i_predicted[:].flatten(),2))

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
#         i_predicted=updatedensity_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
#         trajs_opt.append(i_predicted)
#     trajs_opt = np.copy(np.array(trajs_opt))

#     for i in range(ND):
#         plt.scatter(range(t0,t1),i_actual[:,i],color=CB_color_cycle[i],label=index[i])
#         plt.plot(range(t0+tfit0,t0+tfit1),np.mean(trajs_opt,axis=0)[:tfit1-tfit0,i],color=CB_color_cycle[i])
#         plt.plot(range(t0+tfit1-1,t0+tfit0+t_predict),np.mean(trajs_opt,axis=0)[tfit1-tfit0-1:,i],'--',color=CB_color_cycle[i])
#     plt.ylim(np.min(i_actual),np.max(i_actual)*1.2)
    
#     plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'-',color='black',label='fit')
#     plt.plot([t0+tfit0,t0+tfit1],[-1000,-1000],'--',color='black',label='prediction')
#     plt.title('k='+str(k_opt)+', sigma='+str(sigma_opt)+'\n '+str(tfit1-tfit0)+' timepoints are used in fitting')
#     plt.ylabel('Density of '+focal_variant)
#     plt.legend()
#     plt.savefig('fig/selection/densityspace'+filename+'.pdf',bbox_inches='tight')  
#     plt.show()
    


# def fit_sweep_freq(focal_variant,index, freq_actual, res_A, t0,t1, tfit0,tfit1, round_max=10,filename='demo',k_fixed='n',sigma_fixed ='n'):
#     Path('fig/selection/').mkdir(parents=True, exist_ok=True)
#     err_min=100000000
    
#     nonzeromin =[]
#     aux = i_actual.flatten()
#     for i in aux:
#         if i>0:
#             nonzeromin.append(i)
#     nonzeromin =np.min(nonzeromin)
            
#     ini_old = np.copy(freq_actual[tfit0])
#     for idx, i in enumerate(ini_old):
#         if i<nonzeromin:
#             ini_old[idx] =nonzeromin

    
#     ini_old = np.copy(freq_actual[tfit0])
#     k_old =1.0
#     sigma_old = 0
#     k_res=[]
#     sigma_res=[]
#     err_res=[]
    
#     ND = len(res_A[0])
#     itermax = len(res_A)
#     for rd in range(round_max):
        
#         if k_fixed =='n':
#             k= k_old*np.exp(np.random.normal(0,0.05))
#         else:
#             k =k_fixed
        
#         ini = ini_old*[np.exp(np.random.normal(0,0.05)) for i in range(ND)]
        
#         if sigma_fixed =='n':
#             sigma = sigma_old + np.random.normal(0,0.1)
#             if sigma<-1:
#                 sigma=-1
#         else:
#             sigma = sigma_fixed
            
#         err=0

#         for iter in range(itermax):
#             A = np.copy(res_A[iter])
#             freq_predicted=updatefreq_ode(ini,tfit0,tfit1,A,k, sigma)
#             err+= np.sum(np.power(freq_actual[tfit0:tfit1].flatten() - freq_predicted[:].flatten(),2))

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
#         freq_predicted=updatefreq_ode(ini_opt,tfit0,tfit0+t_predict,A,k_opt, sigma_opt)
#         trajs_opt.append(freq_predicted)
#     trajs_opt = np.copy(np.array(trajs_opt))

#     for i in range(ND):
#         plt.scatter(range(t0,t1),freq_actual[:,i],color=CB_color_cycle[i],label=index[i])
#         plt.plot(range(t0+tfit0,t0+tfit1),np.mean(trajs_opt,axis=0)[:tfit1-tfit0,i],color=CB_color_cycle[i])
#         plt.plot(range(t0+tfit1-1,t0+tfit0+t_predict),np.mean(trajs_opt,axis=0)[tfit1-tfit0-1:,i],'--',color=CB_color_cycle[i])
#     plt.ylim(0,1)
#     plt.plot([t0+tfit0,t0+tfit1],[10,10],'-',color='black',label='fit')
#     plt.plot([t0+tfit0,t0+tfit1],[10,10],'--',color='black',label='prediction')
#     plt.title('k='+str(k_opt)+', sigma='+str(sigma_opt)+'\n '+str(tfit1-tfit0)+' timepoints are used in fitting')
#     plt.ylabel('Frequency of '+focal_variant)
#     plt.legend()
#     plt.savefig('fig/selection/freqspace'+filename+'.pdf',bbox_inches='tight')  
#     plt.show()






def calc_Lvec_Rvec_eval(A):
    ND = len(A)

    Leval, Levec=LA.eig(A.T)  
    idx = np.real(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]

    Reval, Revec=LA.eig(A)  
    idx = np.real(Reval).argsort()[::-1]  #The largest appears the left-most.
    Reval= Reval[idx] # Make sure the descending ordering
    Revec= Revec[:,idx]

    for i in range(ND):
        norm =Levec[:,i]@Revec[:,i]
        Levec[:,i] *=1.0/norm

    # Check the ordering of eigen vectors
    for i in range(ND):
        if np.abs(Reval[i]-Leval[i])>1e-4:
            print('eigenvalues are incorrectly orderd')

    # Check the normalization 
    LR = Levec.T@Revec
    for i in range(ND):
        for j in range(ND):
            if i!=j and np.abs(LR[i,j])>1e-4:
                print('Normalization incorrect')
            elif (i==j and np.abs(LR[i,j]-1)>1e-4):
                print('Normalization incorrect')
            
    return Levec, Revec, Reval

def sol_linear_sweeping(t, freq_ini , Levec, Revec, Reval, k, s):
    res=0
    ND = len(freq_ini)
    for i in range(ND):
        coef = freq_ini @Levec[:,i]*np.exp((s+ k*Reval[i])*t)
        #print('coef',coef)
        res+=coef*Revec[:,i]
    
    if np.sum(np.imag(res))>1e-4:
        print('solution is imaginary')
        
    return np.real(np.round(res,8))