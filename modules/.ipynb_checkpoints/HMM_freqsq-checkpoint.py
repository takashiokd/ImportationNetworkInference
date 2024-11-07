import cvxpy as cp
import pandas as pd
import math
from numpy import linalg as LA
import numpy as np
import random
from numpy.linalg import inv as minv
from modules.HMMtools import *


def calc_freq(counts,totcounts):
    freq= counts.copy()
    for i in range(counts.shape[0]):
        for t in range(counts.shape[2]):
            freq[i,:,t] *=1.0/totcounts[i,t]
    return freq

def mat_flatind(mat):
    shape = mat.shape
    rows, cols = np.indices(shape)
    aux = rows.flatten(), cols.flatten()
    return aux

def freqSQ_mat_to_arr_allcomp(X2mat):#freqSQ_allmat_to_arr(X2mat):
    res=[]
    for t in range(X2mat.shape[2]):
        mat = X2mat[:,:,t].copy()
        res.append(mat[mat_flatind(mat)])
        
    res=np.array(res)
    
    return res.transpose()

def from_A_to_O_allcomp(A):
    ND = len(A)
    Osize = ND*ND
    O = np.zeros((Osize,Osize))
    aux = np.zeros((ND,ND))
    map_Oindex_Aindices = mat_flatind(aux)
    for I in range(Osize):
        i1 = map_Oindex_Aindices[0][I]
        i2 = map_Oindex_Aindices[1][I]
        for J in range(Osize):
            j1 = map_Oindex_Aindices[0][J]
            j2 = map_Oindex_Aindices[1][J]
            O[I,J]+=A[i1,j1] *A[i2,j2]

    return O

# def from_Oind_Aind(ND):
#     aux = np.zeros((ND,ND))
#     map_Oindex_Aindices = mat_flatind(mat)
def LS_X2_allcomp(counts,totcounts,A_start=None,updateA_mode='shift_two',runmax=100):
    #parameters:
    # A
    # Neff
    # Csn
    # denom_sig
    # denom_gam
    
    #reg_factor=100
    if updateA_mode in ['shift_two','nonrev']:
        print(updateA_mode)
    
    res=dict()
    
    ND = counts.shape[0]
    X2mat=calc_freqSQ(counts,totcounts)
    Nsample = np.mean(totcounts,axis=1)
    
    A_LS=calc_ALS(counts, counts_deme=totcounts)
    
    if A_start is None:
        print('ALS')
        A_old=A_LS.copy()
    else:
        print('A, Specified')
        A_old=A_start.copy()


    X2arr=freqSQ_mat_to_arr_allcomp(X2mat) # Narr x T
    
    erropt=1000000
    for run in range(runmax):
        #Update parameters:
        if updateA_mode=='shift_two':
            A_new=update_A(A_old,h=0.0025)
        elif updateA_mode=='nonrev':
            A_new=nonrev_update(A_old)

        AA = from_A_to_O_allcomp(A_new)

        err =0
        for t in range(X2arr.shape[1]-1):
            err+=np.sum((X2arr[:,t+1]-AA@X2arr[:,t])**2)
        
        if erropt >err:
            A_old = A_new.copy()
            erropt=err
        #print(erropt)
    return A_old



#################

def calc_freqSQ(counts,totcounts):
    freq=calc_freq(counts,totcounts)

    Xsq=np.zeros((freq.shape[0],freq.shape[0],freq.shape[2]))
    for i in range(counts.shape[0]):
        for j in range(counts.shape[0]):
            for t in range(counts.shape[2]):
                Xsq[i,j,t]=np.mean(freq[i,:,t]*freq[j,:,t])
                
    return Xsq
                
            
def freqSQ_arr_to_mat(X2arr):
    if len(X2arr.shape)==2:
        ND = int(0.5*(-1 +np.sqrt(1+8*len(X2arr))))
        res=[]
        for t in range(X2arr.shape[1]):
            aux =np.zeros((ND,ND))
            aux[np.triu_indices_from(aux,k=0)] = X2arr[:,t]
            res.append(aux +aux.transpose()-np.diag(np.diag(aux)))
            
    elif len(X2arr.shape)==1:
        print('1')
        ND = int(0.5*(-1 +np.sqrt(1+8*len(X2arr))))
        aux =np.zeros((ND,ND))
        aux[np.triu_indices_from(aux,k=0)] = X2arr[:]
        res=aux +aux.transpose()-np.diag(np.diag(aux))
        
    return np.array(res).transpose()

def freqSQ_mat_to_arr(X2mat):
    res=[]
    for t in range(X2mat.shape[2]):
        mat = X2mat[:,:,t].copy()
        res.append(mat[np.triu_indices_from(mat,k=0)])
        
    res=np.array(res)
    
    return res.transpose()

def from_A_to_O(A):
    ND = len(A)
    Osize = int((ND*ND+ND)/2)
    O = np.zeros((Osize,Osize))
    aux = np.zeros((ND,ND))
    map_Oindex_Aindices = np.triu_indices_from(aux,k=0)
    for I in range(Osize):
        i1 = map_Oindex_Aindices[0][I]
        i2 = map_Oindex_Aindices[1][I]
        for J in range(Osize):
            j1 = map_Oindex_Aindices[0][J]
            j2 = map_Oindex_Aindices[1][J]
            if j1!=j2:
                O[I,J]+=A[i1,j1] *A[i2,j2]
                O[I,J]+=A[i1,j2] *A[i2,j1]
            elif j1==j2:
                O[I,J]+=A[i1,j1] *A[i2,j2]

    return O

def symmat_to_uppertrig(mat):
    return mat[np.triu_indices_from(mat,k=0)]

def uppertrig_to_symmat(arr):
    ND = int(0.5*(-1 +np.sqrt(1+8*len(arr))))
    aux =np.zeros((ND,ND))
    aux[np.triu_indices_from(aux,k=0)] = arr[:]
    return aux +aux.transpose()-np.diag(np.diag(aux))

def arr_labels(ND):
    aux = []
    for i in range(ND):
        for j in range(i,ND):
            aux.append(str(i)+','+str(j))
    return aux


##############################

def LS_F2(F_X1_X0,F_X0_X0):
    results=[]
    n=F_X1_X0.shape[0]
    for i in range(n):

        # Define and solve the CVXPY problem.
        a = cp.Variable(n)
        B=np.zeros(n)
        for k in range(n):
            B[k] = F_X1_X0[i,k] -  F_X1_X0[i,i]
        M=np.zeros((n,n))
        for j in range(n):
            for k in range(n):
                M[j,k] = F_X0_X0[j,k] - F_X0_X0[j,i]

        #Rescaling for numerical stability
        rescale=np.mean([np.mean(B),np.mean(M)])
        B*=1/rescale
        M*=1/rescale
        
        cost = cp.sum_squares(B - a@M)
        prob = cp.Problem(cp.Minimize(cost),[a>=1e-6,np.ones(n) @ a==1])
        prob.solve()
        results.append(a.value)
        
    return np.array(results)
    

def A_Ne_from_F2(counts, totcounts, correction):
    freq=calc_freq(counts,totcounts)

    totcounts_extended  = np.array([totcounts]*freq.shape[1]).transpose([1,0,2]) # make totalcounts the same shape as freq
    unbiased_hetero = freq*(1-freq)*(totcounts_extended/(totcounts_extended-1))  # Patterson et al

    X0=freq[:,:,:-1]
    X1=freq[:,:,1:]

    unbiased_hetero0 = unbiased_hetero[:,:,:-1]
    unbiased_hetero1 = unbiased_hetero[:,:,1:]
    totcounts_extended0 = totcounts_extended[:,:,:-1]
    totcounts_extended1 = totcounts_extended[:,:,1:]

    F_X1_X0=np.zeros((freq.shape[0],freq.shape[0]))
    F_X0_X0=np.zeros((freq.shape[0],freq.shape[0]))
    for i in range(freq.shape[0]):
        for j in range(freq.shape[0]):
            if correction==True:
                F_X1_X0[i,j]= np.mean((X1[i]-X0[j])**2 - unbiased_hetero0[i]/totcounts_extended0[i]- unbiased_hetero1[j]/totcounts_extended1[j])
                if i!=j:
                    F_X0_X0[i,j]= np.mean((X0[i]-X0[j])**2 - unbiased_hetero0[i]/totcounts_extended0[i]- unbiased_hetero0[j]/totcounts_extended0[j])
                else:
                    F_X0_X0[i,j]= np.mean((X0[i]-X0[j])**2)
                    
            elif correction==False:
                
                F_X1_X0[i,j]= np.mean((X1[i]-X0[j])**2)
                F_X0_X0[i,j]= np.mean((X0[i]-X0[j])**2)

    #print(unbiased_hetero0)
    
    A =  LS_F2(F_X1_X0,F_X0_X0)
    
    Neff =np.zeros(freq.shape[0])
    aux1 = A@F_X0_X0
    aux2 = A@F_X0_X0@A.transpose()
    hetero_mean =np.mean(np.mean(unbiased_hetero0,axis=1),axis=1)
    for i in range(freq.shape[0]):
        Neff[i]  = 1/((F_X1_X0[i,i] - aux1[i,i] + 0.5*aux2[i,i])/hetero_mean[i])
        
    if np.min(Neff)<0:
        print('Error: Neff < 0')
    return A,Neff
    
    
def LS_X2(counts,totcounts,A_start=None,updateA_mode='shift_two',runmax=100):
    #parameters:
    # A
    # Neff
    # Csn
    # denom_sig
    # denom_gam
    
    #reg_factor=100
    if updateA_mode in ['shift_two','nonrev']:
        print(updateA_mode)
    
    res=dict()
    
    ND = counts.shape[0]
    X2mat=calc_freqSQ(counts,totcounts)
    Nsample = np.mean(totcounts,axis=1)
    
    A_LS=calc_ALS(counts, counts_deme=totcounts)
    
    if A_start is None:
        print('ALS')
        A_old=A_LS.copy()
    else:
        print('A, Specified')
        A_old=A_start.copy()


    X2arr=freqSQ_mat_to_arr(X2mat) # Narr x T
    
    erropt=1000000
    for run in range(runmax):
        #Update parameters:
        if updateA_mode=='shift_two':
            A_new=update_A(A_old,h=0.0025)
        elif updateA_mode=='nonrev':
            A_new=nonrev_update(A_old)

        AA = from_A_to_O(A_new)

        err =0
        for t in range(X2arr.shape[1]-1):
            err+=np.sum((X2arr[:,t+1]-AA@X2arr[:,t])**2)
        
        if erropt >err:
            A_old = A_new.copy()
            erropt=err
        #print(erropt)
    return A_old



###############

def calc_ALS(counts, counts_deme):
    
    B = (counts.copy()).transpose([2,1,0])
    T, Nlins, ND=B.shape
    
    for i in range(ND):
        for t in range(T):
            if counts_deme[i,t]>0:
                B[t,:,i]*=1.0/counts_deme[i,t]
                

    Ne_old=np.array([1000]*ND)
    
    A_LS=lindyn_qp(B,lam=0)
    return A_LS

def calc_PI(A):
    Leval, Levec=LA.eig(A.T)  
    idx = np.abs(Leval).argsort()[::-1]  #The largest appears the left-most.
    Leval= Leval[idx] # Make sure the descending ordering
    Levec= Levec[:,idx]
    PI=np.abs(Levec[:,0])/sum(abs(Levec[:,0]))
    return PI


#################
# # Forward algorithm of Kalman filter with noises whose expectation values are nonzero
def filter_initial_gen(x0, mu_pre, b_sample,V_pre,  Sigma):
    M = len(Sigma)
    I = np.identity(M)
    
    # initial step
    K0 = V_pre@minv( (V_pre) + Sigma)
    
    mu0 = mu_pre + K0@(x0-mu_pre-b_sample)
    V0 =(I-K0)@V_pre 
    
    normal_mean = mu_pre+b_sample
    normal_cov =  V_pre + Sigma
    lnc_0= logGauss(x0, normal_mean, normal_cov)
    return mu0, V0, lnc_0

def filter_later_gen(x_n, mu_old, V_old, F, b, b_sample,  Sigma,Gamma):
    M = len(Sigma)
    I = np.identity(M)
    
    P_old = F@V_old@F.T + Gamma
    K_n = P_old@minv( (P_old) + Sigma)
    mu_n = (F@ mu_old +b)+ K_n @(x_n-b_sample-F@mu_old-b) 
    V_n =(I-K_n)@P_old 

    normal_mean = F@mu_old+b + b_sample
    normal_cov =  P_old + Sigma
    lnc_n = logGauss(x_n, normal_mean, normal_cov)

    return mu_n, V_n, P_old, lnc_n 

def Kfilter_gen(x, mu_star, V_star, F, b, b_sample, Sigma, Gamma):
    #x:time series of vector ( ND times T)
    #mu_star, V_star: the initial hidden-state distribuion
    #Op:evolution matrix (ND times ND)
    #C:deterministic realtion between hidden states and observables, here, C=I.
    #Nsample: controls the noise in emission
    #Ne: controls the noise in hedden-state transition 
    
    ND = len(F)
    lnLH=0
    mu_filter=[]
    V_filter=[]
    P_filter=[]
    Ginv_filter=[]
    
    mu, V, lnc_0=filter_initial_gen(x[:,0], mu_star, b_sample, V_star, Sigma)
   
    lnLH+=lnc_0
    
    mu_filter.append(mu.copy())
    V_filter.append(V.copy())
    
    freqave=np.mean(x)
    #print(freqave)
    for t in range(1,len(x[0])):
        Amu=F@mu
        
        mu, V, P, lnc_n= filter_later_gen(x[:,t], mu, V, F, b, b_sample,Sigma,Gamma)
        lnLH+=lnc_n
        
        mu_filter.append(mu.copy())
        V_filter.append(V.copy())
        P_filter.append(P.copy())
     
    P_filter.append(F@V@F.T + Gamma.copy())
    return np.array(mu_filter),np.array(V_filter), np.array(P_filter),  lnLH

def logGauss(x, mu, cov):
    vec=np.array([x-mu])
    lamlist=LA.eigvals(cov)
    sumloglam=0
    for i in  lamlist:
        sumloglam+=np.log(abs(i))
    k = len(x)
    logGauss=(-0.5 *vec @ LA.inv(cov) @ vec.T)[0,0] - 0.5*k*np.log(2*3.1415926535)  -0.5*sumloglam

    return logGauss
     
def symmetric_proposal(x,dx, xmin,xmax):
    x_new = x + np.random.normal(0,dx)
    if x_new < xmin:
        x_new = xmin + (xmin-x_new)
    elif x_new >xmax:
        x_new = xmax - (x_new-xmax)
    return x_new

# # UPDATE parameters
def nonrev_update(A):
    r,c= np.random.choice(len(A), 2,replace=False)
    delta = np.random.uniform(-A[r,r],A[r,c])
    A_updated=A.copy()
    A_updated[r,c] -=delta
    A_updated[r,r] +=delta
    return A_updated


def shift_between_two(x,h):
    x_new= np.copy(x)
    pos1,pos2 = random.sample(range(len(x)),k=2)
    L = x[pos1]+x[pos2]
    r = x[pos1]/L
    dr=np.random.normal(0,h)
    if r + dr > 1:
        r = 2 - r - dr
    elif r+ dr<0:
        r = - r - dr
    else: 
        r += dr
    x_new[pos1] = L*r
    x_new[pos2] = L*(1-r)
    return x_new


def update_A(A,h):
    res=np.zeros(A.shape)
    for i in range(len(A)):
        res[i] = shift_between_two(A[i],h)
    return res


def calc_det_bias(Neff,Het):
    ND=len(Neff)
    mat_b = np.zeros((ND,ND))
    for i in range(ND):
        mat_b[i,i] = Het[i]/Neff[i]
    
    return symmat_to_uppertrig(mat_b)

def calc_sample_bias(Nsample,Csn,Het):
    ND=len(Csn)
    mat_b_sample = np.zeros((ND,ND))
    for i in range(ND):
        mat_b_sample[i,i] = Csn[i]*Het[i]/Nsample[i]
    
    return symmat_to_uppertrig(mat_b_sample)


def update_Neff(Neff_old):
    # eps=0.01
    # aux = np.array([i *np.exp(np.random.normal(0,eps)) for i in Neff_old])
    # for i in range(len(aux)):
    #     nmin=100
    #     nmax=1000000
    #     if aux[i]<nmin:
    #         aux[i] = nmin+ (nmin-aux[i])
    #     elif aux[i]>nmax:
    #          aux[i] = nmax- (aux[i]-nmax)
    # return aux

    return np.array([symmetric_proposal(x,dx=100, xmin=100,xmax=10000000) for x in Neff_old])

def update_Csn(Csn_old):
    return np.array([symmetric_proposal(x,dx=0.01, xmin=1,xmax=3) for x in Csn_old])


def calc_sigma(HetHet, HetXX,Nsample,Csn,Ntraj):
    ND = len(Nsample)
    aux =np.zeros((ND,ND))
    idx1,idx2=np.triu_indices_from(aux,k=0)
    Narr=len(idx1)
    sigma = np.zeros((Narr,Narr))
    
    Neffsample=Nsample/Csn
    for I in range(Narr):
        i=idx1[I]
        j=idx2[I]
        for J in range(i,Narr):
            k=idx1[J]
            l=idx2[J]
            
            if i==k:
                sigma[I,J]+=HetXX[i,j,l]/Neffsample[i]
                if j==l:
                    sigma[I,J]+=HetHet[i,j]/(Neffsample[i]*Neffsample[j])
            if i==l:
                sigma[I,J]+=HetXX[i,j,k]/Nsample[i]
                if j==k:
                    sigma[I,J]+=HetHet[i,j]/(Neffsample[i]*Neffsample[j])
            if j==k:
                sigma[I,J]+=HetXX[j,i,l]/Neffsample[j]
            if j==l:
                sigma[I,J]+=HetXX[j,k,i]/Neffsample[j]
    sigma*=1.0/Ntraj
    return sigma

def calc_gamma(HetHet, HetAXAX,Neff,Ntraj):
    ND = len(Neff)
    aux =np.zeros((ND,ND))
    idx1,idx2=np.triu_indices_from(aux,k=0)
    Narr=len(idx1)
    gamma = np.zeros((Narr,Narr))
    
    for I in range(Narr):
        i=idx1[I]
        j=idx2[I]
        for J in range(i,Narr):
            k=idx1[J]
            l=idx2[J]
            
            if i==k:
                gamma[I,J]+=HetAXAX[i,j,l]/Neff[i]
                if j==l:
                    gamma[I,J]+=HetHet[i,j]/(Neff[i]*Neff[j])
            if i==l:
                gamma[I,J]+=HetAXAX[i,j,k]/Neff[i]
                if j==k:
                    gamma[I,J]+=HetHet[i,j]/(Neff[i]*Neff[j])
            if j==k:
                gamma[I,J]+=HetAXAX[j,i,l]/Neff[j]
            if j==l:
                gamma[I,J]+=HetAXAX[j,k,i]/Neff[j]
    gamma*=1.0/Ntraj
    return gamma
    
def MCMC_X2(counts,totcounts,A_start=None,Ne_start=None,tmax_mcmc=100, Qsave=True,outdir='demo',outfilename='',updateA_mode='shift_two', update_Csn='n'):
    #parameters:
    # A
    # Neff
    # Csn
    # denom_sig
    # denom_gam
    
    #reg_factor=100
    if updateA_mode in ['shift_two','nonrev']:
        print(updateA_mode)
    
    res=dict()
    
    ND = counts.shape[0]
    X2mat=calc_freqSQ(counts,totcounts)
    Nsample = np.mean(totcounts,axis=1)
    Ntraj=counts.shape[1]
    
    A_LS=calc_ALS(counts, counts_deme=totcounts)
    
    if A_start is None:
        print('ALS')
        A_old=A_LS.copy()
    else:
        print('A, Specified')
        A_old=A_start.copy()
        
    if Ne_start is None:
        Neff_old=np.array([1000]*ND)
    else:
        print('Ne, Specified')
        Neff_old=Ne_start.copy()

    X2arr=freqSQ_mat_to_arr(X2mat) # Narr x T
 
    Narr=X2arr.shape[0]
    
    freq=calc_freq(counts, totcounts)# ND x Nlin x T 

    freq_reshape=freq.reshape((freq.shape[0],freq.shape[1]*freq.shape[2]))
    
    Afreq_reshape = (A_old@freq_reshape).copy()
    Het = np.mean(freq_reshape*(1-freq_reshape),axis=1)
    
    Het_ini = np.zeros(ND)
    for i in range(ND):
        Het_ini[i]= np.mean(freq[i,:,0]*(1-freq[i,:,0]))
        
    HetXX=np.zeros((ND,ND,ND))
    HetAXAX=np.zeros((ND,ND,ND))
    for i in range(ND):
        for j in range(ND):
            for k in range(ND):
                HetXX[i,j,k] +=np.sum(freq_reshape[i]*(1-freq_reshape[i])*freq_reshape[j]*freq_reshape[k])
                HetAXAX[i,j,k]+=np.sum(freq_reshape[i]*(1-freq_reshape[i])*Afreq_reshape[j]*Afreq_reshape[k])
                
    HetXX*=1.0/freq_reshape.shape[1]
    HetAXAX*=1.0/freq_reshape.shape[1]
    
    HetHet = np.zeros((ND,ND))
    for i in range(ND):
        for j in range(ND):
            HetHet[i,j] +=np.sum(freq_reshape[i]*(1-freq_reshape[i])*freq_reshape[j]*(1-freq_reshape[j]))
    HetHet*=1.0/freq_reshape.shape[1]
    
        
    LogLH_old=-1000000
    Csn=np.ones(ND)       
    Csn_old=np.array([1]*ND)
    
    Sigma=calc_sigma(HetHet=HetHet, HetXX=HetXX,Csn=Csn_old,Nsample=Nsample,Ntraj=Ntraj)
    
    accept=0
    
    #Store the markov process at 100 points. 
    t_out = [t  for t in range(tmax_mcmc)]
    if len(t_out)>1000:
        t_out = t_out[:: int(len(t_out)/1000)]

    #Define the variables to store
    A_mcmc=[]
    Csn_mcmc=[]
    Neff_mcmc=[]
    LogLH_mcmc=[]
    
    b_mcmc=[]
    b_sample_mcmc=[]
    
    Sigma_mcmc=[]
    Gamma_mcmc=[]
    
    acc_mcmc=[]


 #   V_star= np.diag([np.var(X2arr[:,0])]*Narr) 
    #Sigma=calc_sigma(denom_sig_old,denom_sigoff_old,hetero,Ntraj)
    Gamma=calc_gamma(HetHet, HetAXAX,Neff_old,Ntraj)
    b=calc_det_bias(Neff_old,Het)
    b_sample=calc_sample_bias(Nsample,Csn_old,Het)

    b_sample_ini=calc_sample_bias(Nsample,Csn_old,Het_ini)
    mu_star=X2arr[:,0].copy()-b_sample_ini # Unbiased estimate of the true Cij for a given observed Cij=X2arr[:,0]
    V_star=calc_sigma(HetHet=HetHet, HetXX=HetXX,Csn=Csn_old,Nsample=Nsample,Ntraj=Ntraj)

    # b*=0
    # b_sample*=0
    
     #print(b,b_sample)
    #print(X2arr.shape)
   

    # print(Gamma)
    for t_mcmc in range(tmax_mcmc):
        
        #Update parameters:
        if updateA_mode=='shift_two':
            A_new=update_A(A_old,h=0.0025)
        elif updateA_mode=='nonrev':
            A_new=nonrev_update(A_old)
            
        Afreq_reshape = (A_new@freq_reshape).copy()
        
        HetAXAX=np.zeros((ND,ND,ND))
        for i in range(ND):
            for j in range(ND):
                for k in range(ND):
                    HetAXAX[i,j,k]+=np.sum(freq_reshape[i]*(1-freq_reshape[i])*Afreq_reshape[j]*Afreq_reshape[k])
        HetAXAX*=1.0/freq_reshape.shape[1]

        # b & Gamma
        Neff_new=update_Neff(Neff_old)
        #Neff_new=Neff_old.copy()
        b=calc_det_bias(Neff_new,Het)
        
        Gamma=calc_gamma(HetHet, HetAXAX,Neff_new,Ntraj)
        
        # b_sample & Sigma
        if update_Csn=='y':
            Csn_new=update_Csn(Csn_old)
        else:
            Csn_new=np.array([1]*ND)
        b_sample=calc_sample_bias(Nsample,Csn_new,Het)
        
        Sigma=calc_sigma(HetHet=HetHet, HetXX=HetXX,Csn=Csn_new,Nsample=Nsample,Ntraj=Ntraj)
        
        LogLH_new=0
        
        AA = from_A_to_O(A_new)
        
        # b*=0
        # b_sample*=0
       
        mu_filter, V_filter,P_filter, lnLH=Kfilter_gen(x=X2arr, mu_star=mu_star, V_star=V_star, F=AA, b=b,b_sample=b_sample,Sigma=Sigma, Gamma=Gamma)  
        LogLH_new+=lnLH
        
        # acceptance_prob = min(1, np.exp(Delta)). if-else is used for numerical stability
        Delta =(LogLH_new - LogLH_old)
        if Delta>0:
            acceptance_prob=1
        else:
            acceptance_prob = np.exp(Delta)

        # update the state
        r = np.random.random()
        
        if acceptance_prob>r:
            A_old =  A_new.copy()
            Csn_old =  Csn_new.copy()
            Neff_old =  Neff_new.copy()            
            LogLH_old= LogLH_new
            accept+=1
            
            acc_aux=1
        else:
            acc_aux=0
            
        #Store the state
        if t_mcmc in t_out:
            A_mcmc.append(A_old.copy())
            Neff_mcmc.append(Neff_old.copy())
            Csn_mcmc.append(Csn_old.copy())
            LogLH_mcmc.append(LogLH_old)
            b_mcmc.append(np.diag(uppertrig_to_symmat(b.copy())))
            b_sample_mcmc.append(np.diag(uppertrig_to_symmat(b_sample.copy())))
            Sigma_mcmc.append(Sigma.flatten())
            Gamma_mcmc.append(Gamma.flatten())
            acc_mcmc.append(acc_aux)
            
    print('Pacc = ', round(accept/tmax_mcmc*100),'%')
    
    res['tout']=t_out
    res['LogLH'] = LogLH_mcmc
    res['A'] = np.array(A_mcmc)
    res['b'] = np.array(b_mcmc)
    res['b_sample']=np.array(b_sample_mcmc)
    res['Csn'] = np.array(Csn_mcmc)
    res['Neff']=np.array(Neff_mcmc)
    res['Sigma']=np.array(Sigma_mcmc)
    res['Gamma']=np.array(Gamma_mcmc)
    res['ALS']=np.array(A_LS.copy())
    res['acc']=np.array(acc_mcmc.copy())
    
    if Qsave==True:
        outdir='X2_MCMC/'+outdir+'/'
        Path(outdir).mkdir(parents=True, exist_ok=True)
        np.save(outdir+'A'+outfilename, A_mcmc)
        np.save(outdir+'ALS'+outfilename, A_LS)
        np.save(outdir+'b'+outfilename, b_mcmc)
        np.save(outdir+'b_sample'+outfilename, b_sample_mcmc)
        np.save(outdir+'Csn'+outfilename,Csn_mcmc)
        np.save(outdir+'Neff'+outfilename,Neff_mcmc)
        np.save(outdir+'acc'+outfilename,acc_mcmc)

    return res