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
import matplotlib.colors as mcolors

from datetime import datetime
from datetime import timedelta

from modules.variables import CB_color_cycle,dict_region_abb,dict_regionabb_number_England
from modules.LDS import lindyn_qp, Kalman_EM, update_A
import matplotlib.patches as mpatches 
from matplotlib.collections import PatchCollection

import matplotlib as mpl

def CI(data,alpha):
    sortdata=np.sort(data)
    return [sortdata[round(0.5*alpha*len(data))],sortdata[-round(0.5*alpha*len(data))]]

from scipy.linalg import null_space


from shapely.geometry import Polygon
from matplotlib.patheffects import withStroke

from modules.tools import *

from sklearn.manifold import MDS


England_region_index =['North East',  'North West', 'Yorkshire and The Humber', 'East Midlands','West Midlands', 'East of England', 'London','South East', 'South West']
# # Plots


def plot_post_HMM_LS_EM(res_A_mcmc, res_A_LS,res_A_EM, index,filename,outpath='fig/'):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    bins=np.linspace(0,1,25)
    ND = len(index)
    
    res_A_mcmc_rand = np.array(random.choices(res_A_mcmc, k=5000))
    if len(res_A_LS )>0:
        A_LS_res_rand = np.array(random.choices(res_A_LS, k=5000))
    if len(res_A_EM)>0:
        A_EM_res_rand = np.array(random.choices(res_A_EM, k=5000))
    
    fig, axes = plt.subplots(ncols=ND, nrows=ND,figsize=(17,17))
    for r in range(ND):
        for c in range(ND):
            axes[r,c].hist(res_A_mcmc_rand[:,r,c],bins=bins,density=True,label='MCMC')
            if len(res_A_LS )>0:
                axes[r,c].hist(A_LS_res_rand[:,r,c],bins=bins,density=True,label='LS')
            #axes[r,c].hist(A_LSWO_res_rand[:,r,c],bins=bins,density=True,label='LSWO')   
            if len(res_A_EM)>0:
                axes[r,c].hist(A_EM_res_rand[:,r,c],bins=bins,density=True,label='EM')

            axes[r,c].legend()
            axes[r,c].set_xlim(0,1)
            axes[r,c].set_ylim(0,30)
            if r<ND-1:
                axes[r,c].set_xticks([])
            if c>0:
                axes[r,c].set_yticks([])
            if c==0:
                axes[r,c].set_ylabel(index[r],fontsize=8)
            if r==ND-1:
                axes[r,c].set_xlabel(index[c],fontsize=8)
    plt.savefig(outpath+filename+'/'+'A_HMM_LS_EM_'+filename+'.pdf',bbox_inches='tight')  
    plt.show() 


def plot_mat_heatmap_with_diag(mat_mean,mat_low,mat_up, plt_title, index,filename,outpath='fig/',figsize=None, vm=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)

    kw={"labels":index,"fontsize":12,"rotation":90}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    
    if figsize!=None:
        fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))
    
    ND = len(index)
    mat_label=make_txt_heatmap(mat_mean,mat_up,mat_low)  
    
    if vm !=None:
        ax=sns.heatmap(mat_mean,square=True, annot=mat_label,vmin=vm[0],vmax=vm[1], fmt='', cmap="YlGnBu",cbar=True)
    else:
        ax=sns.heatmap(mat_mean,square=True, annot=mat_label, fmt='', cmap="YlGnBu",cbar=True)
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=15)
    plt.ylabel('TO',fontsize=15)
    plt.title(plt_title,fontsize=15)
    plt.savefig(outpath+filename+'/'+'Aheatmap_withdiag_'+filename+'.pdf',bbox_inches='tight')  
    plt.show()  
 
    
# def plot_mat_heatmap_offdiag(adj, mat_mean,mat_low,mat_up,vmax,vmin, plt_title, index, mode, outfile,figsize=None):
    
#     def square(x_cell,y_cell):
#         cl='red'
#         delta=0.025
#         plt.hlines(y=y_cell+delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
#         plt.hlines(y=y_cell+1-delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
#         plt.vlines(x=x_cell+delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)
#         plt.vlines(x=x_cell+1-delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)

#     maxlen = max([len(i) for i in index])
#     if maxlen >10:
#         rot_x = 90
#     else:
#         rot_x =0
#     kw={"labels":index,"fontsize":12,"rotation":rot_x}
#     kw_y={"labels":index,"fontsize":12,"rotation":0}
    
#     if figsize!=None:
#         fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))
    
#     ND = len(index)
#     mat_label=make_txt_heatmap(mat_mean,mat_up,mat_low,  mode)  
    
#     mat_mean_offdiag=np.copy(mat_mean)
#     for i in range(ND):
#         mat_mean_offdiag[i,i]=-0.01
        
#     cmap = copy.copy(plt.get_cmap("YlGnBu"))
#     cmap.set_under('lightgray')
    
   
#     ax=sns.heatmap(mat_mean_offdiag, annot=mat_label,linewidth=0.3, fmt='',cmap=cmap, cbar=True,vmax=vmax,vmin=vmin
#                    #,annot_kws={"fontsize":12}
#                   )
#     plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
#     plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
#     plt.xlabel('FROM',fontsize=15)
#     plt.ylabel('TO',fontsize=15)
#     plt.title(plt_title,fontsize=15)    
#     for pair in adj:
#         ind1=index.index(pair[0])
#         ind2=index.index(pair[1])
#         square(x_cell=ind1,y_cell=ind2)
#         square(x_cell=ind2,y_cell=ind1)
        
#     # use matplotlib.colorbar.Colorbar object
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=15)        
       
#     plt.savefig(outfile,bbox_inches='tight')  
#     plt.show() 

def plot_mat_heatmap_offdiag(adj, mat_mean,mat_low,mat_up,vmax,vmin, plt_title, index,filename,outpath='fig/',figsize=None, Qerr='with_err',title=None, figshow=True):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    def square(x_cell,y_cell):
        cl='red'
        delta=0.025
        plt.hlines(y=y_cell+delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
        plt.hlines(y=y_cell+1-delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
        plt.vlines(x=x_cell+delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)
        plt.vlines(x=x_cell+1-delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)

    maxlen = max([len(i) for i in index])
    if maxlen >10:
        rot_x = 90
    else:
        rot_x =0
    kw={"labels":index,"fontsize":20,"rotation":rot_x}
    kw_y={"labels":index,"fontsize":20,"rotation":0}
    
    if figsize!=None:
        fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))
    
    ND = len(index)
    
    if ND ==8:
        colors = ['#377eb8',
                         '#ff7f00',
                         #'#4daf4a',
                         '#f781bf',
                         '#a65628',
                         '#984ea3',
                         '#999999',
                         '#e41a1c',
                         '#dede00']
    else:
        colors=CB_color_cycle
    
    mat_mean_offdiag=np.copy(mat_mean)
    for i in range(ND):
        mat_mean_offdiag[i,i]= -1 #np.min(mat_mean)-0.01
        
    cmap = copy.copy(plt.get_cmap("YlGnBu"))
    cmap.set_under('Darkgray')
    
    if vmax==None:
        vmax = np.max(mat_mean_offdiag)
   
    ax=sns.heatmap(mat_mean_offdiag,square=True, # annot=mat_label,
                   linewidth=0.3, fmt='',cmap=cmap, cbar=True,vmax=vmax,vmin=vmin
                  , cbar_kws={'shrink':0.8})

    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=20)
    plt.ylabel('TO',fontsize=20)
    plt.title(plt_title,fontsize=20)    
    for pair in adj:
        ind1=index.index(pair[0])
        ind2=index.index(pair[1])
        square(x_cell=ind1,y_cell=ind2)
        square(x_cell=ind2,y_cell=ind1)
        
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()

    for i, label in enumerate(xticklabels):
        label.set_color(colors[i % len(colors)])

    for i, label in enumerate(yticklabels):
        label.set_color(colors[i % len(colors)])

    vmean=0.5*(vmax+vmin)
    for r in range(ND):
        for c in range(ND):
            if mat_mean[r,c] >vmean:
                cl='white'
            else:
                cl ='black'
                
            if Qerr=='with_err':
                ax.annotate(str(np.round(mat_mean[r,c],2)), (c+0.25, r+0.5), fontsize=12, color=cl)
                ax.annotate('['+str(np.round(mat_low[r,c],2))+', '+str(np.round(mat_up[r,c],2))+']', (c+0.03, r+0.8), fontsize=8, color=cl)
            elif Qerr=='without_err':
                ax.annotate(str(np.round(mat_mean[r,c],2)), (c+0.1, r+0.6), fontsize=15, color=cl)
             
    if figshow==True:
        plt.savefig(outpath+filename+'/'+'Aheatmap'+filename+'.pdf',bbox_inches='tight')  
        plt.show()


def plot_Csn(res_Csn, index,xax_label,filename,outpath='fig/',figsize=None,vminmax=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    mpl.style.use('default')
   
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if 'product' in filename:
        aux =int(len(index)/2)
        colors = CB_color_cycle[0:aux]+CB_color_cycle[0:aux]
    else:
        colors = CB_color_cycle
    
    df=pd.DataFrame(res_Csn,columns=index)
    bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=colors,showfliers = False)

    maxlen = max([len(i) for i in index])
    if maxlen >10:
        plt.xticks(rotation=90)
        
    bp.set_xlabel(xax_label, fontsize=15)
    bp.set_ylabel('Sampling noise strength, '+r"$C_{\rm sn}$", fontsize=15)
    
    if vminmax is None:
        plt.ylim(0,np.max(res_Csn)*1.2)
    else:
        plt.ylim(vminmax[0],vminmax[1])
        
    plt.hlines(1,xmin=-1, xmax=len(index)+1,color='gray')
    bp.set_xticklabels(labels=index,fontsize=13)
    plt.xlim(-0.5,len(index)-0.5)
    plt.yticks(fontsize=15)
    plt.grid(True) 
    plt.savefig(outpath+filename+'/'+'Csn'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    
    
def plot_Ne(Ne_data, index,xax_label,filename,outpath='fig/',figsize=None,vminmax=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    mpl.style.use('default')
   
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if 'product' in filename:
        aux =int(len(index)/2)
        colors = CB_color_cycle[0:aux]+CB_color_cycle[0:aux]
    else:
        colors = CB_color_cycle
    
    df=pd.DataFrame(Ne_data,columns=index)
    bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=colors,showfliers = False)

    
    maxlen = max([len(i) for i in index])
    if maxlen >10:
        plt.xticks(rotation=90)
        
    plt.yscale('log')
    bp.set_xlabel(xax_label, fontsize=15)
    bp.set_ylabel('Effective population size, '+r"$N_{\rm e}$", fontsize=15)
    
    if vminmax is None:
        if int(np.log10(np.min(Ne_data)))==int(np.log10(np.max(Ne_data))):
            int_y_min=int(np.log10(np.min(Ne_data)))
            int_y_max=int(np.log10(np.max(Ne_data)))+1
            plt.ylim( np.power(10,int_y_min),np.power(10,int_y_max))
        else:
            plt.ylim( np.min(Ne_data),np.max(Ne_data)*2)
    else:
        plt.ylim(vminmax[0],vminmax[1])
    bp.set_xticklabels(labels=index,fontsize=13)
    
    plt.yticks(fontsize=15)
    plt.grid(True) 
    plt.savefig(outpath+filename+'/'+'Ne'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    

def plot_Ne_Npositive_Npop(res_Ne,random_positive_cases,N_pop, ew,width, index, filename, outpath='fig/',figsize=None, vminmax=None):

    
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)

    ND = len(N_pop)
    data1 = res_Ne/N_pop
    data2 = random_positive_cases/N_pop
    
    if figsize is None:
        fig, ax1 = plt.subplots(figsize=(5,5))
    months = index
    #ax1.set_xlabel('Region')
    ax1.set_ylabel('Ratio', color='k',fontsize=15)
    res1 = ax1.boxplot(
        data1, positions = np.arange(ND)-0.15, widths=0.25,
        patch_artist=True,showfliers = False
    )
    lst_element=['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
    for element in lst_element:
        plt.setp(res1[element], color='k')
    for patch in res1['boxes']:
        patch.set_facecolor('tab:blue')

  
    res2 = ax1.boxplot(
        data2, positions = np.arange(ND)+0.15, widths=0.25,
        patch_artist=True,showfliers = False
    )
    for element in lst_element:
        plt.setp(res2[element], color='k')
    for patch in res2['boxes']:
        patch.set_facecolor('tab:orange')

    ax1.set_xlim([-0.5, ND-0.5])
    
    if vminmax is None:
        if np.min(data2)>0:
            ax1.set_ylim(0.5*np.min([np.min(data1),np.min(data2)]),20*np.max([np.max(data1),np.max(data2)]))
        else:
            ax1.set_ylim(0.5*np.min(data1),10*np.max(np.max(data1)))
    else:
        ax1.set_ylim(vminmax[0],vminmax[1])
            
    ax1.set_yscale('log')
    ax1.set_xticks(np.arange(ND))
    ax1.set_xticklabels(months, rotation=0)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.legend([res1["boxes"][0], res2["boxes"][0]],[r'$N_{{\rm e},i}/N_{{\rm pop},i}$', r'$N_{{\rm positive},i}/N_{{\rm pop},i}$'], loc='upper right',fontsize=15)

    plt.savefig(outpath+filename+'/'+'Ne_Npositive_Npop_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    
    plt.show()
    
    
# def plot_PI(res_PI_vec, index, xax_label,filename,vmax=None,outpath='fig/',figsize=None):
#     Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
 

#     if figsize!=None:
#         plt.figure(figsize=(figsize[0],figsize[1]))
#     if 'product' in filename:
#         aux =int(len(index)/2)
#         colors = CB_color_cycle[0:aux]+CB_color_cycle[0:aux]
#     else:
#         colors = CB_color_cycle
        
#     df=pd.DataFrame(res_PI_vec,columns=index)
    
#     bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette=colors, showfliers = False)
#     bp.set_xticklabels(labels=index,fontsize=8)
#     bp.set_xlabel('', fontsize=0)
    
    
#     maxlen = max([len(i) for i in index])
#    # if maxlen >10:
#     plt.xticks(rotation=90)

#     if vmax is None:
#         vmax = np.max(res_PI_vec)
        
#     bp.set_ylabel("Fixation probability", fontsize=8)
#     plt.yticks(fontsize=8)
#     plt.grid(True) 
#     plt.ylim(0,vmax)
#     plt.savefig(outpath+filename+'/'+'PI'+filename+'.pdf', dpi=300, bbox_inches='tight')  
#     plt.show()

def plot_PI(res_PI_vec, index, xax_label,filename,vmax=None,outpath='fig/',figsize=None,title=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
 

    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if 'product' in filename:
        aux =int(len(index)/2)
        colors = CB_color_cycle[0:aux]+CB_color_cycle[0:aux]
    else:
        colors = CB_color_cycle
        
    if len(index)==8:
        colors = ['#377eb8',
                     '#ff7f00',
                     #'#4daf4a',
                     '#f781bf',
                     '#a65628',
                     '#984ea3',
                     '#999999',
                     '#e41a1c',
                     '#dede00']
        
    df=pd.DataFrame(res_PI_vec,columns=index)
    df_melted = pd.melt(df)
 
    df_melted['hue'] = df_melted['variable']  # Copy 'variable' to 'hue' for unique colors

    # Plot
    plt.figure(figsize=(figsize[0],figsize[1])) if figsize is not None else plt.figure()
    bp = sns.boxplot(x="variable", y="value", hue="hue", data=df_melted, palette=colors, showfliers=False)
    if title is not None:
        bp.set_title(title)
    plt.xticks(ticks=range(len(index)), labels=index, fontsize=8)
    # bp.set_xticklabels(labels=index,fontsize=8)
    bp.set_xlabel('', fontsize=0)
    
    maxlen = max([len(i) for i in index])
   # if maxlen >10:
    plt.xticks(rotation=90)

    if vmax is None:
        vmax = np.max(res_PI_vec)
        
    bp.set_ylabel("Fixation pr.", fontsize=8)
    
    plt.yticks(fontsize=8)
    plt.grid(True) 
    plt.ylim(0,vmax)

        
    plt.savefig(outpath+filename+'/'+'PI'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()

    
    
def plot_PI_norm(res_PI_vec, index, xax_label,filename,vmax=None,outpath='fig/',figsize=None,title=None,ylabel="Normed fixation pr."):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
 

    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if 'product' in filename:
        aux =int(len(index)/2)
        colors = CB_color_cycle[0:aux]+CB_color_cycle[0:aux]
    else:
        colors = CB_color_cycle
        
    if len(index)==8:
        colors = ['#377eb8',
                     '#ff7f00',
                     #'#4daf4a',
                     '#f781bf',
                     '#a65628',
                     '#984ea3',
                     '#999999',
                     '#e41a1c',
                     '#dede00']
        
    df=pd.DataFrame(res_PI_vec,columns=index)
    df_melted = pd.melt(df)
 
    df_melted['hue'] = df_melted['variable']  # Copy 'variable' to 'hue' for unique colors

    # Plot
    plt.figure(figsize=(figsize[0],figsize[1])) if figsize is not None else plt.figure()
    bp = sns.boxplot(x="variable", y="value", hue="hue", data=df_melted, palette=colors, showfliers=False)
    if title is not None:
        bp.set_title(title)
    plt.xticks(ticks=range(len(index)), labels=index, fontsize=8)
    # bp.set_xticklabels(labels=index,fontsize=8)
    bp.set_xlabel('', fontsize=0)
    
    maxlen = max([len(i) for i in index])
   # if maxlen >10:
    plt.xticks(rotation=90)

    if vmax is None:
        vmax = np.max(res_PI_vec)
        
    bp.set_ylabel(ylabel, fontsize=8)
    
    plt.yticks(fontsize=8)
    plt.grid(True) 
    plt.ylim(0,vmax)

        
    plt.savefig(outpath+filename+'/'+'PInorm'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()

    
def plot_spectrum(res_spectrum, filename,outpath='fig/',figsize= None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
        
    ND = len(res_spectrum[0])
    df=pd.DataFrame(res_spectrum,columns=range(1,ND+1))
    df_melted = pd.melt(df)
    df_melted['hue'] = df_melted['variable']
    
    bp=sns.boxplot(x="variable", y="value", hue="hue", data=df_melted,palette="YlOrBr_r",showfliers = False)
   
    bp.set_xlabel("Rank", fontsize=15)
    
    bp.legend_.remove()
    # Explicitly set x-ticks based on ND
    if ND < 10:
        tick_labels = [r'$\lambda_{}$'.format(i + 1) for i in range(ND)]
    else:
        tick_labels = [r'{}'.format(i) if i % 10 == 0 else '' for i in range(ND)]

    # Set x-ticks and x-tick labels here
    bp.set_xticks(range(ND))  # This explicitly sets the x-ticks to match the number of variables
    bp.set_xticklabels(labels=tick_labels, fontsize=15)

    plt.xticks(rotation=0)    
  
    bp.set_ylabel("|Eigenvalue| of " + r"$A$", fontsize=15)
    plt.grid() 
    plt.ylim(0.3,1.05)
    plt.yticks(fontsize=15)
    plt.savefig(outpath+filename+'/'+'spectrum'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    

def plot_nb_vs_nonnb(res_A_mcmc,index, adj, filename,outpath='fig/',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
        
    ND = len(index)
    nb=[]
    non_nb=[]
    for i in range(ND):
        for j in range(i+1,ND):      
            if [index[i], index[j]] in adj or [index[j], index[i]] in adj:
                nb.append(res_A_mcmc[:,i,j])
                nb.append(res_A_mcmc[:,j,i])
            else:
                non_nb.append(res_A_mcmc[:,i,j])
                non_nb.append(res_A_mcmc[:,j,i])

    nb = np.array(nb).flatten()
    non_nb = np.array(non_nb).flatten()

    plt.hist(nb,100,density=True,color='red',histtype='step',label='Neighboring '+r'$ij$')
    plt.hist(non_nb,100,density=True,color='blue',histtype='step',label='Non-neighboring '+r'$ij$')
    plt.hist(nb,100,density=True,color='red',alpha=0.02)
    plt.hist(non_nb,100,density=True,color='blue',alpha=0.02)
    

    plt.yscale('log')
    plt.xlim(0, max(np.max(nb),np.max(non_nb)) )
    plt.ylim(0.01,)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r'$A_{ij}$',fontsize=15)
    plt.ylabel('Posterior distribution, '+r'$ p(A_{ij})$',fontsize=15)
    plt.legend(fontsize=20)
    plt.savefig(outpath+filename+'/'+'Adist_NB_vsNonNB'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    
# def plot_relax(res_A_mcmc,adj, index, filename,outpath='fig/',out_values='Y',figsize=None):
#     Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
#     if figsize!=None:
#         plt.figure(figsize=(figsize[0],figsize[1]))
    
#     nb_index  =[]
#     for pair in adj:
#         nb_index.append([ index.index(pair[0]), index.index(pair[1])])
#         nb_index.append([ index.index(pair[1]), index.index(pair[0])])
#     nb_index =np.array(nb_index)
    
#     res_A_mcmc_drop_nonnb=np.copy(res_A_mcmc)
#     drop_frac=1.0
#     for iter in range(len(res_A_mcmc)):
#         for pair in nb_index:
#             res_A_mcmc_drop_nonnb[iter,pair[0],pair[0]] += drop_frac*res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]]
#             res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]] = (1-drop_frac)*res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]]

#     res_PI_vec_drop_nonnb, res_spectrum_drop_nonnb = calc_PI_spectrum(res_A_mcmc_drop_nonnb)
#     res_PI_vec, res_spectrum = calc_PI_spectrum(res_A_mcmc)
    
#     t_relax= [  1/(1- np.abs(i)) for i in res_spectrum[:,1] ]
#     t_relax_drop_nonnb= [  1/(1- np.abs(i)) for i in res_spectrum_drop_nonnb[:,1] ]

#     df =pd.DataFrame()
#     df['t_relax']= t_relax
#     df['t_relax_drop_nonnb']=t_relax_drop_nonnb
#     bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="YlOrBr_r",showfliers = False)
#     bp.set_xlabel(" ", fontsize=0.)
#     bp.set_xticklabels(labels=['Full',"Only neighboring couplings"],fontsize=15)
#     bp.set_ylabel(r"Relaxation time, $1/(1-|\lambda_1|)$ [week]", fontsize=15)
    
#     if out_values =='Y':
#         mean= np.round(np.mean(t_relax),1)
#         [low,up] =np.round(CI(t_relax,alpha=0.05),1)
#         str1=str(mean)

#         mean= np.round(np.mean(t_relax_drop_nonnb),1)
#         [low,up] =np.round(CI(t_relax_drop_nonnb,alpha=0.05),1)
 
#         str2=str(mean)
        
        
#     plt.title(str1+'     vs     '+ str2,fontsize =20) 
    
#     plt.savefig(outpath+filename+'/'+'relax_NBvsNonNB_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
#     plt.show()    
    
def plot_relax(res_A_mcmc,adj, index, filename,outpath='fig/',out_values='Y',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    colors = ['skyblue', 'lightgreen'] 
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    
    nb_index  =[]
    for pair in adj:
        nb_index.append([ index.index(pair[0]), index.index(pair[1])])
        nb_index.append([ index.index(pair[1]), index.index(pair[0])])
    nb_index =np.array(nb_index)
    
    res_A_mcmc_drop_nonnb=np.copy(res_A_mcmc)
    drop_frac=1.0
    for iter in range(len(res_A_mcmc)):
        for pair in nb_index:
            res_A_mcmc_drop_nonnb[iter,pair[0],pair[0]] += drop_frac*res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]]
            res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]] = (1-drop_frac)*res_A_mcmc_drop_nonnb[iter,pair[0],pair[1]]

    res_PI_vec_drop_nonnb, res_spectrum_drop_nonnb = calc_PI_spectrum(res_A_mcmc_drop_nonnb)
    res_PI_vec, res_spectrum = calc_PI_spectrum(res_A_mcmc)
    
    # t_relax= [  1/(1- np.abs(i)) for i in res_spectrum[:,1] ]
    # t_relax_drop_nonnb= [  1/(1- np.abs(i)) for i in res_spectrum_drop_nonnb[:,1] ]
    t_relax= [  -1/np.log(np.abs(i)) for i in res_spectrum[:,1] ]
    t_relax_drop_nonnb= [  -1/np.log(np.abs(i)) for i in res_spectrum_drop_nonnb[:,1] ]

    df =pd.DataFrame()
    df['t_relax']= t_relax
    df['t_relax_drop_nonnb']=t_relax_drop_nonnb
    
    box_labels = ['Inferred', 'Only\n neighboring']
    #bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),showfliers = False)
    bplot =plt.boxplot(np.array([t_relax,t_relax_drop_nonnb]).transpose(),
                labels=box_labels,showfliers = False,widths=[.5,.5], patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Relaxation time [week]', fontsize=8)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    
    plt.ylim(0,)
    #plt.yticks([0,20,40,60],[0,20,40,60],fontsize=8)
    plt.xticks(fontsize=8)
#     bp.set_xlabel(" ", fontsize=0.)
#     bp.set_xticklabels(labels=['Inferred',"Only \n Neighboring"],fontsize=15)
#     bp.set_ylabel(r"Relaxation time, $1/(1-|\lambda_1|)$ [week]", fontsize=15)
    
    if out_values =='Y':
        mean= np.round(np.mean(t_relax),1)
        [low,up] =np.round(CI(t_relax,alpha=0.05),1)
        str1=str(mean)

        mean= np.round(np.mean(t_relax_drop_nonnb),1)
        [low,up] =np.round(CI(t_relax_drop_nonnb,alpha=0.05),1)
 
        str2=str(mean)
    plt.title(str1+'     vs    '+ str2,fontsize =10) 

    plt.savefig(outpath+filename+'/'+'relax_NBvsNonNB_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show() 
    
def plot_relax_single(res_A_mcmc,filename,outpath='fig/',out_values='Y',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    
    res_PI_vec, res_spectrum = calc_PI_spectrum((np.array(res_A_mcmc)))
    
    #t_relax= [  1/(1- np.abs(i)) for i in res_spectrum[:,1] ]
    t_relax= [  -1/np.log(np.abs(i)) for i in res_spectrum[:,1] ]
 
    df =pd.DataFrame()
    df['t_relax']= t_relax
    #bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),palette="YlOrBr_r",showfliers = False)
    bp=sns.boxplot(x="variable", y="value", data=pd.melt(df),showfliers = False)

    bp.set_xlabel('')  # This hides the x-label

    bp.set_xticks([0])  # Set the tick for the first position
    bp.set_xticklabels(labels=[' '], fontsize=15)  # Set a single, empty label

    bp.set_ylabel(r"Relaxation time [week]", fontsize=15)
    
    if out_values =='Y':

        mean= np.round(np.mean(t_relax),1)
        [low,up] =np.round(CI(t_relax,alpha=0.5),1)

      #  plt.text(0.25,up*0.95, ' '+str(mean)+'\n [' + str(low )+','+ str(up)+']',fontsize =20)
        plt.title(r'$\tau_{\rm relax}$ = '+str(mean)+' [' + str(low )+','+ str(up)+']',fontsize =20)
        plt.xlim(-0.6,0.6)
        
    plt.savefig(outpath+filename+'/'+ 'relax_single'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()   
    
    


def plot_eigen_circle(res_A, filename,outpath='fig/',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    ND = len(res_A[0])
    Arand = np.random.random((ND,ND))
    for i in range(ND):
        Arand[i]*= 1.0/np.sum(Arand[i])
    for i in range(1000):
        Arand = update_A(Arand,h=0.1)
    Amat = Arand
    plt.figure(figsize=(2,2))
    Leval, Levec=LA.eig(Amat) 
    # for i in Leval:
    #     plt.scatter(i.real,i.imag,color= 'gray',s=5,alpha=0.5)
    # plt.scatter(100,100,color= 'gray',label='Random')
        

    Amat =np.mean(res_A,axis=0)
    Leval, Levec=LA.eig(Amat) 
    for i in Leval:
        plt.scatter(i.real,i.imag,color= 'red',s=5,alpha=0.5)
    plt.scatter(100,100,color= 'red',label='Inferred')
    plt.legend()
    
    circle1 = plt.Circle((0, 0), 1, color='black',fill=False)
    plt.gca().add_patch(circle1)
    plt.ylim(-1.05,1.05)
    plt.xlim(-1.05,1.05)
    plt.yticks([-1,0,1])
    plt.xticks([-1,0,1])
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.savefig(outpath+filename+'/'+'circle_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    
def plot_fineA_heatmap(res_A, whichregion, index, filename, outpath='fig/',figsize=None,title='',vmax=0.1):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    A_mean= np.mean(res_A,axis=0)
    ND = len(A_mean)
    A_nodiag=A_mean.copy()
    for i in range(ND):
        A_nodiag[i,i]=np.nan

    switch=[]
    for i in range(ND-1):
        if whichregion[i]!=whichregion[i+1]:
            switch.append(i+1)
    switch=[0]+switch+[ND]

    fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]),dpi=75)

    sns.set_style("white")
    ax = sns.heatmap(A_nodiag,cmap="YlGnBu",xticklabels=index,yticklabels=index, vmax=vmax,cbar_kws={'label': 'Inferred coupling','shrink':0.7})
    ax.figure.axes[-1].yaxis.label.set_size(12)
    for i in range(len(switch)-1):
        ax.hlines(switch[i],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.hlines(switch[i+1],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.vlines(switch[i],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.vlines(switch[i+1],switch[i],switch[i+1],color='gray',alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('FROM')
    ax.set_ylabel('TO')
    ax.set_title(title)
    plt.savefig(outpath+filename+'/'+'fineA_'+filename+'.pdf',dpi=100, bbox_inches ='tight')
   
    plt.show()
    mpl.style.use('default')
    
def plot_fineA_heatmap_UTLA(dict_region_number,res_A, whichregion, index, filename, plt_title='', outpath='fig/',figsize=None):
    
    
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    ND = len( whichregion)
    '''
    plot offdiagonal
    '''
    A_mean= np.mean(res_A,axis=0)
    A_nodiag=A_mean.copy()
    for i in range(ND):
        A_nodiag[i,i]=np.nan

    switch=[]
    for i in range(ND-1):
            switch.append(i+1)
    switch=[0]+switch+[ND]

    ticks_region=['']*ND
    for region in dict_region_number:
        pos=[]
        for idx,i in enumerate(whichregion):
            if region==i:
                    pos.append(idx)
        ticks_region[int(np.mean(pos))]=region

    for idx, i in enumerate(ticks_region):
        if i=='Yorkshire and The Humber':
           ticks_region[idx]= 'Yorkshire and\n The Humber'

    import seaborn as sns
    sns.set_style("white")
    
    ticks_region_abb = []
    for i in ticks_region:
        if i in dict_region_abb.keys():
            ticks_region_abb.append(dict_region_abb[i])
        else:
            ticks_region_abb.append('')
            
    
    ax = sns.heatmap(A_nodiag,cmap="YlGnBu",xticklabels=ticks_region_abb,yticklabels=ticks_region_abb,cbar_kws={'label': 'Inferred coupling'},vmax=0.08)
    ax.figure.axes[-1].yaxis.label.set_size(12)
    for i in range(len(switch)-1):
        ax.hlines(switch[i],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.hlines(switch[i+1],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.vlines(switch[i],switch[i],switch[i+1],color='gray',alpha=0.8)
        ax.vlines(switch[i+1],switch[i],switch[i+1],color='gray',alpha=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('FROM')
    ax.set_ylabel('TO')
    #ax.set_title('Coupling matrix (offdiagonal components)')
    if plt_title !='':
        ax.set_title(plt_title)
    plt.savefig(outpath+filename+'/'+'fineA_'+filename+'.png',dpi=100,bbox_inches ='tight')
    plt.show()
    
def plot_distance_vs_coupling(res_A, gdf, filename,outpath='fig/', method_mean = 'standard',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    
    A_mean= np.mean(res_A,axis=0)
    gdf["x"] = gdf.centroid.x
    gdf["y"] = gdf.centroid.y
    x=list(gdf['x'])
    y=list(gdf['y'])
    ND = len(A_mean)
    if len(gdf)!=ND:
        print('Size of gdf is not ND')

    dist_mig = []
    for i in range(ND):
        for j in range(ND):
            if i > j:
                if method_mean =='geometric':
                    Amean = np.power(A_mean[i,j]*A_mean[j,i],0.5)
                    str_method = r'$\sqrt{A_{ij}A_{ji}}$'
                elif method_mean =='standard': 
                    Amean = (A_mean[i,j] + A_mean[j,i])*0.5
                    str_method = r'$\frac{A_{ij}+A_{ji}}{2}$'
                elif method_mean =='harmonic': 
                    Amean = 2*A_mean[i,j]*A_mean[j,i]/ (A_mean[i,j] + A_mean[j,i])
                    str_method = r'$(\frac{A^{-1}_{ij} +  A^{-1}_{ji}}{2})^{-1}$'
                    
                
                dist_mig.append([ np.sqrt( (x[i]-x[j])**2 +(y[i]-y[j])**2 ),  Amean])
                    
    dist_mig= np.array(dist_mig)
    dist_mig[:,0] *=1.0/np.max(dist_mig[:,0] )

    df = pd.DataFrame(dist_mig,columns = ['distance', 'mig'])
    
    dist_list = np.linspace(0,0.8,20)
    conditional_mig = np.array([[ 0.5*(dist_list [i]+dist_list [i+1]), np.mean(df[(df['distance']>dist_list [i]) & (df['distance']<=dist_list[i+1])]['mig'])] for i in range(len(dist_list)-1)])
    plt.plot(conditional_mig[:,0],conditional_mig[:,1],color ='red')
    plt.scatter(dist_mig[:,0],dist_mig[:,1],s =0.5)
    plt.ylim(2*np.min(dist_mig[:,1]),0.8*np.max(dist_mig[:,1]))
    plt.xlim(1e-2,1)
    plt.xlabel('Normalized geographical distance',fontsize=15)
    plt.ylabel('Symmetrized coupling, '+str_method ,fontsize=15)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(outpath+filename+'/'+'dist_vs_rate_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    
    
def plot_fineA_between_vs_within(res_A, whichregion, filename,outpath='fig/',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    A_mean= np.mean(res_A,axis=0)
    ND = len(A_mean)
    A_between=[]
    A_within=[]
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                if whichregion[i]==whichregion[j]:
                    A_within.append(A_mean[i,j])
                else:
                    A_between.append(A_mean[i,j])
    numbins=25
    plt.figure(figsize=(5,3))

    plt.hist(A_between, bins=np.linspace(0,0.15,numbins),alpha=0.4,edgecolor='lightblue',density=True,label='Across regions',zorder=1)
    plt.hist(A_within, bins=np.linspace(0,0.15,numbins),color='red',edgecolor='red',alpha=0.5,density=True,label='Within regions',zorder=-2)
    plt.ylabel('Probability density',fontsize=15)
    plt.xlabel('Inferred coupling',fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks([0,0.05,0.1,0.15,0.2],fontsize=15)
    plt.legend(fontsize=15)
    plt.xlim(0,0.2)
    plt.yscale('log')
    plt.savefig(outpath+filename+'/'+'A_within_vs_between_regions_'+filename+'.pdf',bbox_inches ='tight')
    plt.show()
    mpl.style.use('default')
    
def plot_fixprob_spectrum(res_A, index, filename,outpath='fig/',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    res_PI_vec, res_spectrum=calc_PI_spectrum(res_A)
    
    
    ND = len(res_A[0])
    index = np.array(index)
    data= res_spectrum

    ticks =[]
    if ND>80:
        for i in range(ND):
            if i%20==1:
                ticks.append(str(i))
            else:
                ticks.append('')
    elif ND>40:
        for i in range(ND):
            if i%10==1:
                ticks.append(str(i))
            else:
                ticks.append('')
    else:
        for i in range(ND):
              ticks.append(str(i+1))
        

    colors=sns.color_palette("husl", len(data.transpose())).as_hex()
    colors=[ i + '80' for i in colors]
    colors2=[]
    for i in colors:
        colors2.append(i)
        colors2.append(i)

    sns.set_style("white")

    bp=plt.boxplot(data,
                  #vert=False,  # Rotate
                  patch_artist=True,  # Enables detailed settings
                  widths=0.5# box width
                  )
    fs=23
    plt.yticks(fontsize=fs)
    plt.xticks(range(1,ND+1),labels=ticks,fontsize=fs)
    plt.xlabel('Rank',fontsize=fs)
    plt.ylabel(r'$| {\rm Eigenvalue}|$',fontsize=fs)
   # plt.hlines(1., xmin=-10,xmax=ND,color='gray',alpha=0.2)
    #plt.hlines(0.5, xmin=-10,xmax=ND,color='gray',alpha=0.2)
  #  plt.hlines(0.0, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    # plt.hlines(-0.5, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    # plt.hlines(-1, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    
    # for i in range(1,ND,10):
    #     plt.vlines(i, ymin=-5,ymax=5,color='gray',alpha=0.2)
    plt.xlim(0,ND+1)
    plt.grid(True) 
    plt.ylim(-0.0,1.05)
    # box color
    for b, c in zip(bp['boxes'], colors):
        b.set(color=c, linewidth=1)  # box outline
        b.set_facecolor(c) # box color
    # mean, whisker, fliers
    for b, c in zip(bp['means'], colors):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['whiskers'], colors2):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['caps'], colors2):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['fliers'], colors):
        b.set(markeredgecolor=c, markeredgewidth=0)
    plt.savefig(outpath+filename+'/'+'eigenvalues_'+filename+'.pdf',bbox_inches ='tight')
    plt.show()

    reltime=np.round(np.mean(1/(1-res_spectrum[:,1])),1)
    [low,up]=np.round(CI(1/(1-res_spectrum[:,1]),0.318),1)
    std=np.round(np.std(1/(1-res_spectrum[:,1])),1)
    print('Relaxation time = '+str(reltime ), low, up, std)
    with open(outpath+filename+'/'+'relax'+filename+'.txt', 'w') as f:
        f.write('Relaxation time = '+str(reltime ) )
        f.write('\n [ '+ str(low)+ ","+ str(up)+ '] \n std ' +str(std) )     

    # # Plot fixation prob
    
    median=np.median(res_PI_vec,axis=0)
    df=pd.DataFrame(np.array([list(range(ND)),median]).transpose(),columns = ['ID','median'])
    df=df.sort_values(by=['median'],ascending=False)
    ID=[round(i) for i in list(df['ID'])]
    res_PI_vec_sort=res_PI_vec[:,ID]
    df['state'] = index[ID]

    data= res_PI_vec_sort    
    ticks =[]
    for i in range(ND):
        if i%10==1:
            ticks.append(str(i))
        else:
            ticks.append('')

    colors=sns.color_palette("husl", len(data.transpose())).as_hex()
    colors=[ i + '80' for i in colors]
    colors2=[]
    for i in colors:
        colors2.append(i)
        colors2.append(i)

    sns.set_style("white")
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
        
    bp=plt.boxplot(data,
                  #vert=False,  # Rotate
                  patch_artist=True,  # Enables detailed settings
                  widths=0.5# box width
                  )
    plt.xticks(range(1,ND+1),labels=list(df['state']))
    #plt.xlabel('State')
    #plt.xticks(range(ND),labels=ticks)
    #plt.xlabel('Rank')

    plt.ylabel('Fixation probability, '+ r'$\Pi_i$')
    # plt.hlines(0.1, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    # plt.hlines(0.001, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    # plt.hlines(0.00001, xmin=-10,xmax=ND,color='gray',alpha=0.2)
    #plt.ylim(0.0000001,np.max(data))
    for i in range(1,ND,10):
        plt.vlines(i, ymin=-5,ymax=5,color='gray',alpha=0.2)
    plt.xlim(0,ND+1)
    plt.ylim(0,np.max(data))
    maxlen = max([len(i) for i in index])
    if maxlen >10:
        plt.xticks(rotation=90)
    # box color
    for b, c in zip(bp['boxes'], colors):
        b.set(color=c, linewidth=1)  # box outline
        b.set_facecolor(c) # box color
    # median, whisker, fliers
    for b, c in zip(bp['medians'], colors):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['whiskers'], colors2):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['caps'], colors2):
        b.set(color=c, linewidth=1)
    for b, c in zip(bp['fliers'], colors):
        b.set(markeredgecolor=c, markeredgewidth=0)
    plt.grid(True) 
    plt.savefig(outpath+filename+'/'+'fixationprob_'+filename+'.pdf',bbox_inches ='tight')
    plt.show()    
    

    
    
    
def visualize_A_age_group(res_A,th,index, filename, outpath='fig/'):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    A_mean =np.mean(res_A,axis=0)
    th =0.12
    Ngroups=len(index)
    a = 2.5
    b =0.9 
    polys = [ Polygon([(1, 0+i), (a, 0+i), (a, b+i), (1, b+i )]) for i in range(Ngroups)]
    x_left  = [ 1  for i in range(Ngroups)]
    y_left  = [  0.5*(b+2*i) for i in range(Ngroups)]
    x_right  = [ a for i in range(Ngroups)]
    y_right  = [ 0.5*(b+2*i) for i in range(Ngroups)]
    df= pd.DataFrame()
    df['geometry'] = polys
    df['NAME'] = [ i for i in range(Ngroups)]

    polys_gdf = gpd.GeoDataFrame(df, geometry='geometry')

    fig, ax = plt.subplots(figsize=(15,10))

    polys_gdf.plot(ax=ax,color = CB_color_cycle,zorder=3,alpha=0.2)
    polys_gdf.boundary.plot(ax=ax,color='black',linewidth=1,zorder=5,alpha=1)

    for i in range(Ngroups):
        for j in range(Ngroups):
            if A_mean[i,j]>th and i!=j:   

                #Coordinates
                dy=0.20

                if i>j:
                    x_ini,y_ini=x_left[j],y_left[j]-0.5*dy
                    x_end,y_end=x_left[i],y_left[i]+dy
                    ud = 'up'
                    th1=90
                    th2=270
                else:
                    x_ini,y_ini=x_left[j],y_left[j]-0.5*dy
                    x_end,y_end=x_left[i],y_left[i]
                    ud = 'down'
                    th1=90
                    th2=270

                #Line width
                rw=A_mean[i,j]*0.15
                cl=CB_color_cycle[j]


                draw_self_loop(ax, up_or_down=ud, center=(0.5*(x_ini+x_end),  0.5*(y_ini+y_end)), radius= 0.5*np.abs(y_ini-y_end),rwidth= rw,facecolor=cl,edgecolor=cl, theta1=th1, theta2=th2)
    for i in range(Ngroups):
         plt.text(0.5*(x_left[i]+x_right[i])-0.45, y_left[i]-0.1,index[i] ,fontsize=25, color =CB_color_cycle[i],path_effects=[withStroke(linewidth=5, foreground='white')],zorder=5)

    plt.xlim(-a+1,a+0.1)            
    plt.axis("off")
    plt.savefig(outpath+filename+'/'+'A_age_'+filename+'_th'+str(th)+'.png',dpi=100)

    plt.show()  
    
    
    

    
#  # For timeseries in England
def load_timeseries(focal_variant,ND,Nlins, timepoints,window_width,noisemode,itermax,dir_hmm):
    halfwidth=int(np.round(window_width/2)-1)

    A_mean_series=np.zeros((len(timepoints),ND,ND))
    A_up_series=np.zeros((len(timepoints),ND,ND))
    A_low_series=np.zeros((len(timepoints),ND,ND))

    Ne_mean_series=np.zeros((len(timepoints),ND))
    Ne_up_series=np.zeros((len(timepoints),ND))
    Ne_low_series=np.zeros((len(timepoints),ND))


    series=[]
    for idx,timepoint in enumerate(timepoints):

        filename='_'+focal_variant+'_Nlins'+str(Nlins)+'_'+str(timepoint-halfwidth)+'_'+str(str(timepoint-halfwidth+window_width))+'_noisemode'+str(noisemode)
        res_A_mcmc, res_Ne_mcmc= load_MCMC_noplots(dir_hmm=dir_hmm, filename=filename, itermax=itermax, burn_in =0.80)
        A_mean,A_low,A_up=calc_A_mean_low_up(res_A_mcmc,alpha=0.5)
        Ne_mean,Ne_low,Ne_up=calc_Ne_mean_low_up(res_Ne_mcmc,alpha=0.5)

        A_mean_series[idx] = np.copy(A_mean)
        A_up_series[idx] = np.copy(A_up)
        A_low_series[idx] = np.copy(A_low)
        Ne_mean_series[idx] = np.copy(Ne_mean)
        Ne_up_series[idx] = np.copy(Ne_up)
        Ne_low_series[idx] = np.copy(Ne_low)    
    
    return [[A_mean_series, A_up_series,A_low_series],[Ne_mean_series,Ne_up_series,Ne_low_series]]

def plot_Aseries(focal_variant,Nlins, timepoints,A_mean_series, A_up_series, A_low_series,index,  outpath='fig/',figsize=None):
    Path(outpath+'/series/').mkdir(parents=True, exist_ok=True)
    
    ND = len(index)
    fig, axs = plt.subplots(ND,ND,figsize=(20,20),sharex=True,sharey=True)
    for row in range(ND):
        for col in range(ND):
            if row==col:
                cl='red'
            else:
                cl='blue'
            axs[row,col].plot(timepoints,A_mean_series[:,row,col],color=cl)
            axs[row,col].fill_between(timepoints, A_up_series[:,row,col], A_low_series[:,row,col],alpha=0.1,color=cl)

            x=[]
            for idx, i in enumerate(timepoints):
                if idx%4==0:
                    x.append(i)
                else:
                    x.append(False)
            axs[row,col].set_xticks(x)
            axs[row,col].set_ylim(0,1)
            axs[row,col].set_xlim(timepoints[0],timepoints[-1])
            axs[row,col].grid(True)  

            if col==0:
                axs[row,col].set_ylabel(index[row]+'\n \n Inferred coupling',fontsize=15)
            if row==ND-1:
                axs[row,col].set_xlabel('Epiweek \n \n'+index[col],fontsize=15)

    fig.tight_layout()    
    fig.savefig(outpath+'/series/'+'Aseries_'+focal_variant+'_Nlins'+str(Nlins)+'.pdf', dpi=200, bbox_inches='tight')  
    plt.show()   
    

def Englandregions_plot_Neseries(focal_variant,Nlins, timepoints,Ne_mean_series, Ne_up_series, Ne_low_series,index,  outpath='fig/',figsize=None):
    ND = len(index)
    fig, axs = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    mpl.style.use('default')
    x=[]
    
    Ne_data = np.copy(Ne_mean_series)
    if int(np.log10(np.min(Ne_data)))==int(np.log10(np.max(Ne_data))):
        int_y_min=int(np.log10(np.min(Ne_data)))
        int_y_max=int(np.log10(np.max(Ne_data)))+1
        plt.ylim( np.power(10,int_y_min),np.power(10,int_y_max))
    else:
        plt.ylim( np.min(Ne_low_series),np.max(Ne_up_series*2))
        
    for idx, i in enumerate(timepoints):
        if idx%4==0:
            x.append(i)
        else:
            x.append(False)
    for i in range(ND):
        row, col =int(i/3),i%3

        mpl.style.use('default')
        axs[row,col].plot(timepoints,Ne_mean_series[:,i])
        axs[row,col].fill_between(timepoints, Ne_up_series[:,i], Ne_low_series[:,i],alpha=0.1)

        axs[row,col].set_yscale('log')

        #axs[row,col].set_ylim(100,10000)

        axs[row,col].set_xticks(x)
#        axs[row,col].set_yticks([100,1000,10000])
        axs[row,col].set_xlim(timepoints[0],timepoints[-1])

        axs[row,col].set_title(index[i])
        axs[row,col].grid(True)

        if col==0:
            axs[row,col].set_ylabel(r'${ N}_{\rm eff}$',fontsize=15)
        if row==2:
            axs[row,col].set_xlabel('Epiweek',fontsize=15)
            
    fig.tight_layout()    
    fig.savefig(outpath+'/series/'+'Neseries_'+focal_variant+'_Nlins'+str(Nlins)+'.pdf', dpi=200, bbox_inches='tight')  
    plt.show()   
    
    
    

def plot_eigenvectors(res_A, index, filename, outpath='fig/',figsize=None,vmax=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    ND = len(res_A[0])
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
        
    Amat = np.mean(res_A,axis=0)
    Leval, Levec=LA.eig(Amat.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]#location times rank
    Levec= np.abs(Levec)
    for i in range(ND):
        Levec[:,i] *=1/np.sum(Levec[:,i])

        
    xtickslabels=[]
    for i in range(1,ND+1):
        if i%10==1:
            xtickslabels.append(str(i))
        else:
            xtickslabels.append('')
    if ND < 15:
        xtickslabels = ['$\langle {} |$'.format(i) for i in range(ND)]

    if vmax is None:
        s=sns.heatmap(Levec,xticklabels = xtickslabels, yticklabels=index,cmap='viridis')
    else:
        s=sns.heatmap(Levec,xticklabels = xtickslabels, yticklabels=index,cmap='viridis',vmax=vmax)
        
    s.set(xlabel='Left eigenvector', ylabel=None)#,title = 'Left Eigenvectors')
   # plt.xticks(rotation=90)
    plt.savefig(outpath+filename+'/'+'ABS_Left_evecs_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    


    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    Leval, Levec=LA.eig(Amat)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx] #location times rank
    Levec= np.abs(Levec)
    for i in range(ND):
        Levec[:,i] *=1/np.sum(Levec[:,i])

    if vmax is None:
        s=sns.heatmap(Levec,xticklabels = xtickslabels, yticklabels=index,cmap='viridis')
    else:
        s=sns.heatmap(Levec,xticklabels = xtickslabels, yticklabels=index,cmap='viridis',vmax=vmax)
        
    s.set(xlabel='Rank', ylabel=None,title = 'Right Eigenvectors')

   #plt.xticks(rotation=90)
    
    plt.savefig(outpath+filename+'/'+'ABS_Right_evecs_'+filename+'.pdf', dpi=300, bbox_inches='tight')  
    plt.show()
    
    
    
def draw_heatmap_dendrogram(a,title,filename, outpath='fig/',figsize=None,cpad=0.1):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram

    metric = 'euclidean'
    method = 'average'

    main_axes = plt.gca()
    divider = make_axes_locatable(main_axes)

    plt.sca(divider.append_axes("left", 1.0, pad=0.))
    ylinkage = linkage(pdist(a, metric=metric), method=method, metric=metric)
    ydendro = dendrogram(ylinkage, orientation='left', no_labels=True,
                         distance_sort='descending',
                         link_color_func=lambda x: 'black')
    plt.gca().set_axis_off()
    a=a.iloc[ydendro['leaves']]

    plt.sca(main_axes)
    plt.imshow(a, aspect='auto', interpolation='none',
                vmin=np.min(a.to_numpy()), vmax=np.max(a.to_numpy()),cmap='viridis')
    plt.colorbar(pad=cpad)
    plt.title('Contribution to '+ title+' eigenvectors')
    plt.xlabel('Rank')
    plt.gca().yaxis.tick_right()
    plt.xticks(range(a.shape[1]), a.columns, rotation=0, size='small')
    plt.yticks(range(a.shape[0]), a.index, size='small')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    plt.savefig(outpath+filename+'/'+title+'eigenvec_dendogram'+filename+'.jpg', dpi=200, bbox_inches='tight')  
    plt.show()

def plot_eigenvectors_dendrogram(res_A, index, filename, outpath='fig/',figsize=None,cpad=0.):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    ND = len(res_A[0])
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
        
    Amat = np.mean(res_A,axis=0)
    Leval, Levec=LA.eig(Amat.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]#location times rank
    Levec= np.abs(Levec)
    for i in range(ND):
        Levec[:,i] *=1/np.sum(Levec[:,i])

        
    xtickslabels=[]
    for i in range(1,ND+1):
        if i%10==1:
            xtickslabels.append(str(i))
        else:
            xtickslabels.append('')
    if ND < 15:
        xtickslabels = list(range(1,ND+1))
    
    left_eigenvecs= np.copy(Levec)
    right_eigenvecs= np.copy(Levec)
    

    
    a=pd.DataFrame(left_eigenvecs, index=index, columns=xtickslabels)
    draw_heatmap_dendrogram(a,title='left', filename=filename, outpath=outpath,figsize=figsize,cpad=cpad)

    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    Leval, Levec=LA.eig(Amat)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx] #location times rank
    Levec= np.abs(Levec)
    for i in range(ND):
        Levec[:,i] *=1/np.sum(Levec[:,i])

    right_eigenvecs= np.copy(Levec)
    a=pd.DataFrame(right_eigenvecs, index=index, columns=xtickslabels)
    draw_heatmap_dendrogram(a,title='right',filename=filename, outpath=outpath,figsize=figsize,cpad=cpad)
    
    
# # Detalied balance

def check_DB_A(res_A,Ath=0.05,hist_xmax = None,filename='filename', insetlbwh = [0.21, 0.5, 0.24, 0.29], outpath='fig/',figsize=None):
    
    print(filename)
    print('itermax',len(res_A))
    ND = len(res_A)
    res=[]
    for iter in range(len(res_A)):
        aux,pairs = db_ratio(res_A[iter],Ath=Ath)#aux= Aji/Aij ,PIj/PIi for major pairs
        if len(aux)>0:
            res.append(aux) 
        
    
    
    res_numpairs_DBviolatepercent =[]
    for  Aratio_PIratio in res:
        data =  Aratio_PIratio[:,1]/Aratio_PIratio[:,0]
        res_numpairs_DBviolatepercent.append([len(data), round((len(data[data<0.5])+len(data[data>2]))/len(data)*100,1)])
    res_numpairs_DBviolatepercent=np.array(res_numpairs_DBviolatepercent)
    del data
    
    res_mean,pairs=db_ratio(res_A[0],Ath=Ath)#db_ratio(np.mean(res_A,axis=0),Ath=Ath)


    if figsize is not None:
        fig, ax1 = plt.subplots(figsize=(figsize[0],figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=(8,5))

        
    PIA_ratio=res_mean[:,1]/res_mean[:,0] # PI_j Aji/ PI_i Aij, which should be one uder DB        
    data = PIA_ratio.copy()
    if hist_xmax is None:
        hist_xmax = np.max([1.2*np.max(data),10])
    cl1='salmon'
    cl2='deepskyblue'
    ax1.hist(data, bins = 10 **np.linspace(0.8*np.log10(np.min(data)), 1.2*np.log10(np.max(data)), 12),color=cl1)
    ax1.set_xscale("log")
    # ax1.set_xticklabels(fontsize=20)
    # ax1.set_yticklabels(fontsize=20)
    ax1.set_xlabel(r'$\frac{A_{ji}}{A_{ij}} \frac{\Pi_{j}}{\Pi_{i}}$',fontsize=20)
    ax1.set_ylabel('Number of pairs '+'$(i,j)$',fontsize=15)

    ax1.set_xlim([1.0/hist_xmax,hist_xmax])

    left, bottom, width, height = insetlbwh
    
    xmax =np.round(np.max(data))
    
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot([0,xmax],[0,xmax],'--',color=cl2)
    ax2.scatter(res_mean[:,0],res_mean[:,1],color=cl2,s=50,facecolors='none')
    ax2.set_aspect('equal')

    ax2.set_xlim([0,xmax])
    ax2.set_ylim([0,xmax])
    ax2.set_xticks(np.linspace(0,xmax,3))
    ax2.set_yticks(np.linspace(0,xmax,3))
    ax2.set_xlabel(r'$\frac{A_{ij}}{A_{ji}}$',fontsize=20,color=cl2)
    ax2.set_ylabel(r'$\frac{\Pi_{j}}{\Pi_{i}}$',fontsize=20,color=cl2)
    ax2.tick_params(color='green', labelcolor=cl2)
    for spine in ax2.spines.values():
            spine.set_edgecolor(cl2)
    plt.savefig(outpath+filename+'/'+'DBratio_Ath.{}'.format(Ath)+'_'+filename+'.pdf', dpi=250, bbox_inches='tight')  
    plt.show()
    
    
    mean_numpairs = int(np.mean(res_numpairs_DBviolatepercent[:,0]))
    std_numpairs =int(np.std(res_numpairs_DBviolatepercent[:,0]))
    mean_DBviolatepercent = np.round(np.mean(res_numpairs_DBviolatepercent[:,1]),1)
    std_DBviolatepercent =np.round(np.std(res_numpairs_DBviolatepercent[:,1]))
                             
    str_out = '{} pm {} pairs'.format(mean_numpairs,std_numpairs)+' are '+'min[Aij,Aji]>{}'.format(Ath)+ '(strongly-interacting pairs)'
    str_out+='\n For {} pm {}'.format(mean_DBviolatepercent, std_DBviolatepercent) + ' % of strongly-interacting pairs, PIi Aij/PIj Aji is <0.5 or >2'

    
    f = open(outpath+filename+'/'+'DBratio_Ath.{}'.format(Ath)+'_'+filename+'.txt', "w")
    f.write(str_out)
    f.close()
    
    print(str_out)
    return res_numpairs_DBviolatepercent


def check_DB_A_boxplot(res_A, Ath,index, filename='filename', outpath='fig/',figsize=None):

    res=[]
    ND = len(res_A[0])
    
    if ND <10:
        for iter in range(len(res_A)):
            dbr,pairs = db_ratio(res_A[iter],Ath=0)  # Ath=0:db_ratio for all pairs
            res.append(dbr)
        res =np.array(res)

        ratio_A_PI= np.array([ res[:,i,1]/res[:,i,0] for i in range( int(ND*(ND-1)/2) )])

        xlabel = []
        Amean = np.round(np.mean(res_A,axis=0),2)
        
        idx_major_pairs=[]
        xlabel=[]
        for idx,pair in enumerate(pairs):
            pos_comma = pair.find(',')
            j = int(pair[:pos_comma])
            i = int(pair[pos_comma+1:])
            if Amean[i,j]>Ath and Amean[j,i]>Ath:
                idx_major_pairs.append(idx)
                xlabel.append( index[j]+', '+index[i] )
        
        
        ratio_A_PI =ratio_A_PI[idx_major_pairs]
        major_pairs = (np.array(pairs))[idx_major_pairs]
        
        if figsize!=None:
            plt.figure(figsize=(figsize[0],figsize[1]))
        else:
            plt.figure(figsize=(15,5))
        plt.boxplot(np.log10(ratio_A_PI.T), showfliers=False)
        #plt.yscale('log')
        plt.hlines(y=0,xmin=0.5,xmax=len(ratio_A_PI )+0.5)
        plt.ylim(-2,2)
        plt.ylabel(r'${\rm Log}_{10} \ \frac{\Pi_i A_{ij}}{\Pi_j A_{ji}}$',fontsize=20)
        plt.xlabel('Pairs $i,j$\n',fontsize=20)
        plt.xticks(range(1,len(major_pairs )+1),xlabel,rotation=90)
        plt.grid(axis='y')
        plt.savefig(outpath+filename+'/'+'DBratio_BSerror_Ath.{}'.format(Ath)+filename+'.pdf', dpi=250, bbox_inches='tight')
        plt.show()


        

def plot_flux(res_flux ,MorMtilde, group,Tc,Qannot='with_err'):
    
    res_aux=res_flux.copy()
    mean,low,up= calc_A_mean_low_up(res_aux,alpha=0.5)
    a = np.log10(mean)
    sns.set(font_scale=1.1)
    
    if MorMtilde=='M':
        kws={'label':r'$\log_{10} [{M}^{Tc}_{ij}]$','shrink':0.7}
    elif MorMtilde =='Mtilde':
        kws={'label':r'$log_{10} [\tilde{M}^{Tc}_{ij}]$','shrink':0.7}
        
#     if vmax is False:
#         vmax=np.max(a)
#     else:
#         vmax= vmax
        
    
    sns.heatmap(a, 
                vmin=np.min(a), vmax=np.max(a),#np.max(take_offdiag(a)),
                square=True,
                annot=make_txt_heatmap(np.log10(mean), np.log10(up), np.log10(low), mode=Qannot),
                fmt='',xticklabels=group,yticklabels=group,
                cmap="YlGnBu",
                cbar=True,
                cbar_kws=kws)
    
    if MorMtilde=='M':
        plt.title('Flux matrix, '+r'${M}^{Tc}_{ij}$, '+'$T_c ={}$'.format(Tc))
    elif MorMtilde =='Mtilde':
        plt.title(r'${\tilde M}^{Tc}_{ij}=\frac{M^{Tc}_{ij}}{\Pi_i\Pi_j}$, '+'$T_c ={}$'.format(Tc))
    mpl.style.use('default')
        

        

    
def plot_mat_simple(res_A, plt_title, index, Qannot='with_err',vmin=None,vmax=None):
    
    if index is not None:
        kw={"labels":index,"fontsize":12,"rotation":90}
        kw_y={"labels":index,"fontsize":12,"rotation":0}
    else:
        kw={}
        kw_y={}
        
    ND = len(res_A[0])
    
    mean,low,up= calc_A_mean_low_up(res_A,alpha=0.5)
        
    if (vmax is None):
        vmax= np.max(take_offdiag(mean))
    if (vmin is None):
        vmin= 0
        
        
    sns.heatmap(mean,
                square=True,
                annot=make_txt_heatmap(mean, up, low, mode=Qannot),
                vmin=vmin,vmax=vmax, 
                fmt='', cmap="YlGnBu",
                cbar=True, cbar_kws={'shrink':0.7})
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=15)
    plt.ylabel('TO',fontsize=15)
    plt.title(plt_title,fontsize=15)
    mpl.style.use('default')
    
def plot_mat_simple_offdiag(res_A, plt_title, index, Qannot='with_err',vmin=None,vmax=None):
    kw={"labels":index,"fontsize":12,"rotation":0}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    ND = len(res_A[0])
    
    mean,low,up= calc_A_mean_low_up(res_A,alpha=0.5)
        
    if (vmax is None):
        vmax= np.max(take_offdiag(mean))
    if (vmin is None):
        vmin= 0
        
    mean_offdiag= mean.copy()
    for i in range(ND):
        mean_offdiag[i,i]=-1
        
    cmap = copy.copy(plt.get_cmap("YlGnBu"))
    cmap.set_under('Darkgray')
    sns.heatmap(mean_offdiag,
                square=True,
                annot=make_txt_heatmap(mean, up, low, mode=Qannot),
                vmin=vmin,vmax=vmax, 
                fmt='', cmap="YlGnBu",
                cbar=True, cbar_kws={'shrink':0.7})
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=15)
    plt.ylabel('TO',fontsize=15)
    plt.title(plt_title,fontsize=15)
    mpl.style.use('default')
    

def logratio_mat(res_A):
    ND =len(res_A[0])
    ratio_mat = np.zeros((len(res_A),ND,ND))
    for idx, A in enumerate(res_A):
        for i in range(ND):
            for j in range(ND):
                ratio_mat[idx,i,j]=np.log10(A[i,j]/A[j,i])
    return ratio_mat

def linearratio_mat(res_A):
    ND =len(res_A[0])
    ratio_mat = np.zeros((len(res_A),ND,ND))
    for idx, A in enumerate(res_A):
        for i in range(ND):
            for j in range(ND):
                ratio_mat[idx,i,j]=A[i,j]/A[j,i]
    return ratio_mat
                   
        
        
def plot_A_ratio(res_A,  index, Qannot='with_err',vmin=None,vmax=None,outpath='fig/',figsize=None,filename=''):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    kw={"labels":index,"fontsize":12,"rotation":0}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    ND = len(res_A[0])
    
    mean,low,up= calc_A_mean_low_up(logratio_mat(res_A),alpha=0.5)
 
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if (vmax is None):
        vmax= 1.5*np.max(take_offdiag(mean))
    if (vmin is None):
        vmin= 1.5*np.min(take_offdiag(mean))
        
    res=sns.heatmap(mean,
                square=True,
                annot=make_txt_heatmap(mean, up, low, mode=Qannot),
                vmin=vmin,vmax=vmax, 
                fmt='', cmap="bwr",
                cbar=True, cbar_kws={'shrink':0.7,'label':'Log10[Ratio]'})
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM, $j$',fontsize=15)
    plt.ylabel('TO, $i$',fontsize=15)
    plt.title(r'$\log_{10}\,A_{ij}/A_{ji}$',fontsize=15)

#     cbarvals=np.linspace(vmin,vmax,5)
#     cbar = res.collections[0].colorbar
#     cbar.set_ticks(cbarvals)
#     cbar.set_ticklabels([ np.round(np.power(10,i),1) for i in cbarvals])
    
    # Drawing the frame
    res.axhline(y = 0, color='k',linewidth = 3)
    res.axhline(y = mean.shape[1], color = 'k',
                linewidth = 3)

    res.axvline(x = 0, color = 'k',
                linewidth = 3)

    res.axvline(x = mean.shape[0], 
                color = 'k', linewidth = 3)
    plt.savefig(outpath+filename+'/'+'Aratio_'+filename+'.pdf', dpi=250, bbox_inches='tight')
    plt.show()
                 
    mpl.style.use('default')
    
    
def plot_NA_ratio(res_A,  index, Qannot='with_err',vmin=None,vmax=None,outpath='fig/',figsize=None,filename='', Npop=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    kw={"labels":index,"fontsize":12,"rotation":0}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    ND = len(res_A[0])
    
    mean,low,up= calc_A_mean_low_up(logratio_mat(res_A),alpha=0.5)
 
    NpopA = np.zeros((ND,ND))
    for i in range(ND):
        for j in range(ND):
               NpopA[i,j] = mean[i,j]*Npop[i]/Npop[j]
    mean= NpopA.copy()
    
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if (vmax is None):
        vmax= 1.5*np.max(take_offdiag(mean))
    if (vmin is None):
        vmin= 1.5*np.min(take_offdiag(mean))
    if np.abs(vmax)>np.abs(vmin):
        vmin = -1.0*np.abs(vmax)
    else:
        vmax = +1.0*np.abs(vmin)


    res=sns.heatmap(mean,
                square=True,
                annot=make_txt_heatmap(mean, up, low, mode=Qannot),
                vmin=vmin,vmax=vmax, 
                fmt='', cmap="bwr",
                cbar=True, cbar_kws={'shrink':0.7,'label':'Log10[Ratio]'})
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM, $j$',fontsize=15)
    plt.ylabel('TO, $i$',fontsize=15)
    plt.title(r'$\log_{10}\, N_i A_{ij}/N_j A_{ji}$',fontsize=15)
    
#     cbarvals=np.linspace(vmin,vmax,5)
#     cbar = res.collections[0].colorbar
#     cbar.set_ticks(cbarvals)
#     cbar.set_ticklabels([ np.round(np.power(10,i),1) for i in cbarvals])

    # Drawing the frame
    res.axhline(y = 0, color='k',linewidth = 3)
    res.axhline(y = mean.shape[1], color = 'k',
                linewidth = 3)

    res.axvline(x = 0, color = 'k',
                linewidth = 3)

    res.axvline(x = mean.shape[0], 
                color = 'k', linewidth = 3)
    plt.savefig(outpath+filename+'/'+'NpopAratio_'+filename+'.pdf', dpi=250, bbox_inches='tight')
    plt.show()

    mpl.style.use('default')
    
    
def plot_PIA_ratio(res_A, n, index, Qannot='with_err',vmin=None,vmax=None,outpath='fig/',figsize=None,filename=''):


    kw={"labels":index,"fontsize":12,"rotation":0}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    if len(res_A)>1000:
        res_A_rand = np.array(random.choices(res_A, k=1000))
    else:
        res_A_rand = np.copy(res_A)

    ND = int(len(res_A[0]))

    res_mat=[]
    for i in range(len(res_A_rand )):
        Amat = res_A_rand[i]
        Leval, Levec=LA.eig(Amat.T)  
        idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
        Leval= Leval[idx] # Make sure the descending ordering
        Levec= Levec[:,idx]
        PI_vec= Levec[:,0].copy()
        PI_vec=np.abs(PI_vec)/sum(abs(PI_vec))

        An = np.linalg.matrix_power(Amat, n).copy()
        mat =An.copy()
        for i in range(ND):
            for j in range(ND):
                mat[i,j] *=PI_vec[i]
        res_mat.append(mat)

    res_mat= np.array(res_mat)
    mean,low,up= calc_A_mean_low_up(logratio_mat(res_mat),alpha=0.5)

    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    if (vmax is None):
        vmax= 1.5*np.max(take_offdiag(mean))
    if (vmin is None):
        vmin= 1.5*np.min(take_offdiag(mean))

    if np.abs(vmax)>np.abs(vmin):
        vmin = -1.0*np.abs(vmax)
    else:
        vmax = +1.0*np.abs(vmin)

    res=sns.heatmap(mean,
                square=True,
                annot=make_txt_heatmap(mean, up, low, mode=Qannot),
                vmin=vmin,vmax=vmax, 
                fmt='', cmap="bwr",
                cbar=True, cbar_kws={'shrink':0.7,'label':'Log10[Ratio]'})
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM, $j$',fontsize=15)
    plt.ylabel('TO, $i$',fontsize=15)
    plt.title(r'$\Pi_i (A^n)_{ij}/\Pi_j (A^n)_{ji},\ n=$'+str(n),fontsize=15)

    # cbarvals=np.linspace(vmin,vmax,5)
    # cbar = res.collections[0].colorbar
    # cbar.set_ticks(cbarvals)
    # cbar.set_ticklabels([ np.round(np.power(10,i),1) for i in cbarvals])

    # Drawing the frame
    res.axhline(y = 0, color='k',linewidth = 3)
    res.axhline(y = mean.shape[1], color = 'k',
                linewidth = 3)

    res.axvline(x = 0, color = 'k',
                linewidth = 3)

    res.axvline(x = mean.shape[0], 
                color = 'k', linewidth = 3)

    plt.savefig(outpath+filename+'/'+'PIA'+str(n)+'ratio_'+filename+'.pdf', dpi=250, bbox_inches='tight')
    plt.show()

    mpl.style.use('default')

    
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def draw_heatmap_dendrogram_simplever(data, dist,index,figsize):

   
    method = 'average'

    a=pd.DataFrame(data, index=index, columns = [i if i%20==0 else '' for i in range(len(data))])

    plt.figure(figsize=(figsize[0],figsize[1]))
    main_axes = plt.gca()
    divider = make_axes_locatable(main_axes)

    plt.sca(divider.append_axes("left", 1.0, pad=0))
    ylinkage = linkage(dist,metric=metric)
    ydendro = dendrogram(ylinkage, orientation='left', no_labels=True,
                         distance_sort='descending',
                         link_color_func=lambda x: 'black')
    plt.gca().set_axis_off()
    a=a.iloc[ydendro['leaves']]

    plt.sca(main_axes)
    plt.imshow(a, aspect='auto', interpolation='none',
                vmin=np.min(a.to_numpy()), vmax=np.max(a.to_numpy()),cmap='viridis')
    plt.colorbar(pad=0.2)
    plt.title('Contribution to eigenvectors')
    plt.xlabel('Rank')
    plt.gca().yaxis.tick_right()
    plt.xticks(range(a.shape[1]), a.columns, rotation=0, size='small')
    plt.yticks(range(a.shape[0]), a.index, size='small')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    
    plt.show()
    
    
def Aheatmap_vs_demes_England(dict_A_vs_num_per_deme, dict_whichregion_vs_num_per_deme, maintitle=''):
    rows=3
    cols=3
    plt.figure(figsize=(12,12))
    for figidx, num_per_deme in enumerate([1,2,3,4,5,6,7,8]):
        plt.subplot(rows,cols,figidx+1)

        A = dict_A_vs_num_per_deme[num_per_deme]
        whichregion= dict_whichregion_vs_num_per_deme[num_per_deme]
        ND = len(A)
        A_nodiag=A.copy()
        for i in range(ND):
            A_nodiag[i,i]=np.nan

        switch=[]
        for i in range(ND-1):
            if whichregion[i]!=whichregion[i+1]:
                switch.append(i+1)
        switch=[0]+switch+[ND]
        sns.set_style("white")

        ticks_region=['']*ND
        for region in dict_regionabb_number_England:
            pos=[]
            for idx,i in enumerate(whichregion):
                if region==i:
                        pos.append(idx)
            ticks_region[int(np.mean(pos))]=region

        vmax = 0.15

        cbtf=False

        ax = sns.heatmap(A_nodiag,cmap="YlGnBu",cbar=cbtf,xticklabels=ticks_region,yticklabels=ticks_region,cbar_kws={'label': 'Inferred coupling',"shrink": 0.5},vmax=vmax)

        if figidx+1 in [7,8]:
            #ax = sns.heatmap(A_nodiag,cmap="YlGnBu",xticklabels=ticks_region,yticklabels=False,cbar=cbtf,cbar_kws={'label': 'Inferred coupling',"shrink": 0.5},vmax=vmax)
            ax.set_xlabel('FROM')
        if  figidx+1 in [1,4,7]:
            #print('aaa',figidx)
            #ax = sns.heatmap(A_nodiag,cmap="YlGnBu",xticklabels=False,yticklabels=ticks_region,cbar=cbtf,cbar_kws={'label': 'Inferred coupling',"shrink": 0.5},vmax=vmax)
            ax.set_ylabel('TO')

        ax.figure.axes[-1].yaxis.label.set_size(12)
        for i in range(len(switch)-1):
            ax.hlines(switch[i],0,ND,color='gray',alpha=0.3)
            ax.hlines(switch[i+1],0,ND,color='gray',alpha=0.3)
            ax.vlines(switch[i],0,ND,color='gray',alpha=0.3)
            ax.vlines(switch[i+1],0,ND,color='gray',alpha=0.3)
        ax.set_aspect('equal')

        ax.set_title('#demes/region = {}'.format(num_per_deme))

    plt.subplot(rows,cols,9)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="YlGnBu"), label='Coupling strength',pad=0.9,shrink=0.9,location='top')
    plt.gca().set_visible(False)
    #plt.savefig(figdirname+'Autla_CGgeodist_{}.png'.format(ND),dpi=100, bbox_inches = 'tight')    
    plt.suptitle( maintitle)
    plt.subplots_adjust(top=0.95)   
    

def make_txt_mean(mean):
    label=[]
    ND = len(mean)
    for i in range(ND):
        aux = [str(np.round(mean[i,j],2))  for j in range(ND)]
        label.append(aux)
    return np.array(label)

def plot_mat_heatmap_offdiag_simple( mat_mean,vmax,vmin, plt_title, index,ax):

    maxlen = max([len(i) for i in index])
    if maxlen >10:
        rot_x = 90
    else:
        rot_x =0
    kw={"labels":index,"fontsize":13,"rotation":rot_x}
    kw_y={"labels":index,"fontsize":13,"rotation":0}
    
    ND = len(index)
    
    mat_mean_offdiag=np.copy(mat_mean)
    for i in range(ND):
        mat_mean_offdiag[i,i]= -1
    cmap = copy.copy(plt.get_cmap("YlGnBu"))
    cmap.set_under('Darkgray')
    
    if vmax==None:
        vmax = np.max(mat_mean_offdiag)
   
    mat_label = np
    sns.heatmap(mat_mean_offdiag,square=True, annot=make_txt_mean(mat_mean),
                   linewidth=0.3, fmt='',cmap=cmap, cbar=True,vmax=vmax,vmin=vmin
                  , cbar_kws={'shrink':0.5}, ax = ax)
    ax.set_xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    ax.set_yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    ax.set_xlabel('FROM',fontsize=18)
    ax.set_ylabel('TO',fontsize=18)
    ax.set_title(plt_title,fontsize=15)    

    
def plot_asym( mat_mean,vmax,vmin, plt_title, index,ax):

    maxlen = max([len(i) for i in index])
    if maxlen >10:
        rot_x = 90
    else:
        rot_x =0
    kw={"labels":index,"fontsize":13,"rotation":rot_x}
    kw_y={"labels":index,"fontsize":13,"rotation":0}
    
    ND = len(index)
    
    mat_mean_offdiag=np.copy(mat_mean)
    # for i in range(ND):
    #     mat_mean_offdiag[i,i]= -1
    cmap = copy.copy(plt.get_cmap("bwr"))
    #cmap.set_under('Darkgray')
    
    if vmax==None:
        vmax = np.max(mat_mean_offdiag)
   
    mat_label = np
    sns.heatmap(mat_mean_offdiag,square=True, annot=make_txt_mean(mat_mean),
                   linewidth=0.3, fmt='',cmap=cmap, cbar=True,vmax=vmax,vmin=vmin
                  , cbar_kws={'shrink':0.5}, ax = ax)
    ax.set_xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    ax.set_yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    ax.set_xlabel('FROM',fontsize=18)
    ax.set_ylabel('TO',fontsize=18)
    ax.set_title(plt_title,fontsize=18)  
    
    
    
    
    
    
def plt_MDS_deme_lat(demelist,mass_demes,whichregion, dist_matrix, physcoord=None,namedisplayed=None,angle_correction=0,ref='n',figsize=None,title=''):
    index=list(demelist)
    regions = England_region_index
    ND = len(demelist)

    mds = MDS(
        n_components=2,
        max_iter=50000,
        eps=1e-5,
        random_state=10,
        dissimilarity="precomputed",
        n_jobs=1)


    colors = [CB_color_cycle [England_region_index.index(whichregion[i])]  for i in range(ND)] 

    df_demes = pd.DataFrame()
    df_demes['deme']=demelist
    df_demes['region']=whichregion
    df_demes['regionidx']=[ England_region_index.index(re) for re in whichregion]
    df_demes['x']=mass_demes[:,0]
    df_demes['y']=mass_demes[:,1]
    df_demes['idx'] = [i for i in range(len(df_demes))]
    df_demes['namedisplayed']=namedisplayed
    #df_demes= df_demes.sort_values(by = 'y').reset_index()
    cmap=mcolors.ListedColormap(mcolors.LinearSegmentedColormap.from_list("", ["red","white","blue"])(np.linspace(0, 1, len(df_demes))))

    
    
    pos = mds.fit_transform(dist_matrix)
    


    center_south = np.median(pos[list(df_demes[(df_demes['region']=='London') | (df_demes['region']=='South East')| (df_demes['region']=='South West')].index)],axis=0)
    center_north = np.median(pos[list(df_demes[(df_demes['region']=='North East') | (df_demes['region']=='North West')| (df_demes['region']=='Yorkshire and The Humber')].index)],axis=0)
    
    print(center_south,center_north)
#     theta_London = calc_angle(center_London) 
    angle = math.atan2(center_south[1]-center_north[1],center_south[0]-center_north[0])/math.pi*180 -90+angle_correction
    pos_rotated = rotate(pos,angle=angle, reflect='')
    
    center_LDN = np.median(pos[list(df_demes[(df_demes['region']=='London')].index)],axis=0)
    center_SW = np.median(pos[list(df_demes[(df_demes['region']=='South West')].index)],axis=0)
    
    if center_LDN[0]>center_SW[0]:
       
        pos_rotated = rotate(pos_rotated,angle=0, reflect='x')

    if ref=='y':
       
        pos_rotated = rotate(pos_rotated,angle=0, reflect='x')
        
    if figsize!=None:
        plt.figure(figsize=(figsize[0],figsize[1]))
    else:
        fig_x=6*1.5
        fig_y=6*1.5
        fig=plt.figure(figsize=(fig_x,fig_y))
    ax=plt.subplot(111)
    
    
    for i in range(pos_rotated.shape[0]):
        ax.scatter(pos_rotated[i,0],pos_rotated[i,1],s=10,linewidth=1, cmap=cmap, facecolors=CB_color_cycle[df_demes['regionidx'].iloc[i]],alpha=0.9)
        
        xscale= np.max(pos_rotated[:,0])-np.min(pos_rotated[:,0])
        if namedisplayed is not None:
            ax.annotate(df_demes['namedisplayed'].iloc[i], (pos_rotated[i,0]-.5* xscale/6, pos_rotated[i,1]+0.1*xscale/6),color = CB_color_cycle[df_demes['regionidx'].iloc[i]],size=4)
    
    plt.xlim(np.min(pos_rotated[:,0])*1.2,np.max(pos_rotated[:,0])*1.2)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    plt.title(title)
    plt.grid()
   
    
    plt.tight_layout()
    
    
from scipy.spatial.distance import cosine

def kl_divergence(p, q):
    """ Calculate Kullback-Leibler divergence """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def distdemes_from_A(A,mode):
    
    ND =len(A)
    dist_matrix= np.zeros((ND,ND)) #(a+a.T)/2 ### needs to be symmetric

        
    if mode=='log':
        for i in range(ND):
            for j in range(ND):
                if i!=j:
                    dist_matrix[i,j] =1-1.0*np.log10(np.sqrt(A[i,j]*A[j,i]))
                    
                    
    elif mode =='L2':
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                dist_matrix[i,j] = np.sqrt(np.sum((p-q)**2))
                
    elif mode =='LogL2':
        
        
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                dist_matrix[i,j] = np.sqrt(np.sum((np.log(p)-np.log(q))**2))
                
                              
    elif mode =='corr':
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                # re-normalize
                p *=1.0/np.sum(p)
                q *=1.0/np.sum(q)
                
                corr_matrix = np.corrcoef(p,q)
                corr = corr_matrix[1,0]
                dist_matrix[i,j] =1-corr
                
    elif mode =='rankcorr':
        from scipy import stats
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
            
                # re-normalize
                p *=1.0/np.sum(p)
                q *=1.0/np.sum(q)
                
                corr_matrix = stats.spearmanr(p,q)
                corr = corr_matrix[0]
                dist_matrix[i,j] =1-corr

    elif mode =='MFPT':
        M = calc_MFPT(A)
        
        for i in range(ND):
            for j in range(ND):
                dist_matrix[i,j] =np.sqrt(M[i,j]*M[j,i])
                
    elif mode== 'relax_oskar':

        from numpy.linalg import matrix_power
        Apowers=np.array([matrix_power(A,pow) for pow in range(1,100)])
        for i in range(ND):
            for k in range(i+1,ND):
                
                dist_matrix[i,k]=np.sum(np.power(Apowers[:,i,:]-Apowers[:,k,:],2))/np.sum(np.power(A[i,:]-A[k,:],2))
                dist_matrix[k,i]=dist_matrix[i,k]
                
    elif mode== 'relax_oskar_mod':

        from numpy.linalg import matrix_power
        Apowers=np.array([matrix_power(A,pow) for pow in range(1,100)])
        for i in range(ND):
            for k in range(i+1,ND):
                
                dist_matrix[i,k]=np.sqrt(np.sum(np.power(Apowers[:,i,:]-Apowers[:,k,:],2))/np.sum(np.power(A[i,:]-A[k,:],2)))
                dist_matrix[k,i]=dist_matrix[i,k]
                
    elif mode =='corr_Bh':#Bhattacharyya distance
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                # re-normalize
                p *=1.0/np.sum(p)
                q *=1.0/np.sum(q)
                
                dist_matrix[i,j] =-np.log(np.sum(np.sqrt(p*q)))
                
    elif mode =='corr_He':#Hellinger
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                # re-normalize
                p *=1.0/np.sum(p)
                q *=1.0/np.sum(q)
                
                dist_matrix[i,j] =np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)#np.sqrt(1-np.sum(np.sqrt(p*q)))
                
    elif mode =='corr_JS':#JensenShannon
        for i in range(ND):
            for j in range(ND):
                p = A[i].copy()
                q = A[j].copy()
                
                p = np.delete(p,[i,j])
                q = np.delete(q,[i,j])
                
                # re-normalize
                p *=1.0/np.sum(p)
                q *=1.0/np.sum(q)
                m = 0.5*(p+q)
                
                dist_matrix[i,j] =0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
                
                
#         for i in range(ND):
#             for k in range(i+1,ND):
#                 beta=np.sum(np.power(Apowers[:,i,:]-Apowers[:,k,:],2))/np.sum(np.power(A[i,:]-A[k,:],2))
#                 dist_matrix[i,k]=-1./np.log(1.-1./beta)
#                 dist_matrix[k,i]=dist_matrix[i,k]
                
#                 beta=0
#                 count=0
#                 for j in range(ND):
#                     if j!=i and j!=k:
#                         count+=1
#                         beta+=np.sum(power(Apowers[:,i,j]-Apowers[:,k,j],2)/np.power(A[i,j]-A[k,j],2))
#                 beta *=1/count
                
#                 dist_matrix[i,k]=-1./np.log(1.-1./beta)
#                 dist_matrix[k,i]=dist_matrix[i,k]


    return np.sqrt(dist_matrix)
            
