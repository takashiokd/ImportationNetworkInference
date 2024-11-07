#EM algorithm Kalman
from numpy import linalg as LA

import numpy as np
from numpy.linalg import inv as minv
from cvxopt import matrix, solvers
import random
import os

import time

def errA(A,Aref):
    ND = len(A)
    Delta=0
    count=0
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                count+=1
                Delta+=(A[i,j]-Aref[i,j])**2
    return np.sqrt(Delta/count)



def logGauss(x, mu, cov):
    vec=np.array([x-mu])
    lamlist=LA.eigvals(cov)
    sumloglam=0
    for i in  lamlist:
        sumloglam+=np.log(abs(i))
    k = len(x)
    logGauss=(-0.5 *vec @ LA.inv(cov) @ vec.T)[0,0] - 0.5*k*np.log(2*3.1415926535)  -0.5*sumloglam

    return logGauss
    
    
def getdiag(A):
    A_diag=np.array([ A[i,i] for i in range(len(A))])
   
    return A_diag




def filter_initial(x0, mu_pre, V_pre,  Sigma):
    M = len(Sigma)
    I = np.identity(M)
    
    # initial step
    
    if np.linalg.det(V_pre + Sigma)==0:
        print('filter_ini.V_pre',V_pre)
        print('filter_ini.Sigma',Sigma)
    K0 = V_pre@minv( (V_pre) + Sigma)
    
    mu0 = mu_pre + K0@(x0-mu_pre)
    V0 =(I-K0)@V_pre 
    
    normal_mean = mu_pre
    normal_cov =  V_pre + Sigma
    lnc_0= logGauss(x0, normal_mean, normal_cov)
    return mu0, V0, lnc_0

def filter_later(x_n, mu_old, V_old, A,  Sigma,Gamma):
    M = len(Sigma)
    I = np.identity(M)
    
    P_old = A@V_old@A.T + Gamma
    
    if np.linalg.det(P_old+Sigma)==0:
        print("filter_later.A",A)
        print("filter_later.P_old",P_old)
        print("filter_later.Sigma",Sigma)
        
    K_n = P_old@minv( (P_old) + Sigma)
    mu_n = A@ mu_old + K_n @(x_n-A@mu_old)
    V_n =(I-K_n)@P_old 

    normal_mean = A@mu_old
    normal_cov =  P_old + Sigma
    lnc_n = logGauss(x_n, normal_mean, normal_cov)

    return mu_n, V_n, P_old, lnc_n 

    
def Kfilter(x, mu_star, V_star, A,  counts_deme, Ne, Csn, noisemode):
    #x:time series of vector ( ND times T)
    #mu_star, V_star: the initial hidden-state distribuion
    #A:evolution matrix (ND times ND)
    #C:deterministic realtion between hidden states and observables, here, C=I.
    #Nsample: controls the noise in emission
    #Ne: controls the noise in hedden-state transition 
    
    ND = len(A)
    lnLH=0
    mu_filter=[]
    V_filter=[]
    P_filter=[]
    Ginv_filter=[]
    
   
    Sigma = np.diag([ Csn[i]*mu_star[i]*(1-mu_star[i])/counts_deme[i,0]  for i in range(ND)])
 
    mu, V, lnc_0=filter_initial(x[:,0], mu_star, V_star, Sigma)
   
    #lnLH+=lnc_0
    
    mu_filter.append(mu.copy())
    V_filter.append(V.copy())
    
   
    freqave=np.mean(x)
    #print(freqave)
    for t in range(1,len(x[0])):
        #print('t',t)
        Amu=A@mu
        
        if noisemode==0:
            Sigma = np.diag([Csn[i]*x[i,t]*(1-x[i,t])/(counts_deme[i,t])  for i in range(ND)])
            Gamma = np.diag([ x[i,t]*(1-x[i,t])/Ne[i] for i in range(ND)])
        elif noisemode==1:
            Sigma = np.diag([Csn[i]*freqave*(1-freqave)/(counts_deme[i,t])  for i in range(ND)])
            Gamma = np.diag([ freqave*(1-freqave)/Ne[i] for i in range(ND)])
            # Sigma = np.diag([ Amu[i]*(1-Amu[i])/counts_deme[i,t]  for i in range(ND)])
            # Gamma = np.diag([ Amu[i]*(1-Amu[i])/Ne[i] for i in range(ND)])
        elif noisemode==2:
            xaux = np.mean(x,axis=1)
            Sigma = np.diag([Csn[i]* xaux[i]*(1-xaux[i])/(counts_deme[i,t])  for i in range(ND)])
            Gamma = np.diag([ xaux[i]*(1-xaux[i])/Ne[i] for i in range(ND)])
        elif noisemode==3:
            Sigma = np.diag([Csn[i]*x[i,t]*(1-x[i,t])/(counts_deme[i,t])  for i in range(ND)])
            Gamma = np.diag([ x[i,t-1]*(1-x[i,t-1])/Ne[i] for i in range(ND)])
            
        
        mu, V, P, lnc_n= filter_later(x[:,t], mu, V, A,  Sigma,Gamma)
        lnLH+=lnc_n
        
        mu_filter.append(mu.copy())
        V_filter.append(V.copy())
        P_filter.append(P.copy())
     
    P_filter.append(A@V@A.T + Gamma)
    return np.array(mu_filter),np.array(V_filter), np.array(P_filter),  lnLH


def Ksmoother (A, mu_filter, V_filter,P_filter):
    
    T = len(mu_filter)
    ND =len(A)
    mu_smoother=np.zeros((T,ND))
    V_smoother=np.zeros((T,ND,ND))
    J=np.zeros((T,ND,ND))
    
    mu_smoother[-1]=mu_filter[-1]
    V_smoother[-1]=V_filter[-1]
    J[-1] =  V_filter[-1]@A.transpose()@minv(P_filter[-1])
    for t in reversed(range(len(mu_filter)-1)):
        J[t] =  V_filter[t]@A.transpose()@minv(P_filter[t])
        mu_smoother[t] = mu_filter[t] + J[t] @ (mu_smoother[t+1] - A@mu_filter[t] )
        V_smoother[t] = V_filter[t] + J[t]@(V_smoother[t+1] - P_filter[t])@J[t].transpose()
    
    return mu_smoother, V_smoother, J


def Expvals(mu_smoother, V_smoother, J):
    T,ND =mu_smoother.shape
    
    Ez = np.copy(mu_smoother)
    Ezz_pre = np.zeros((T,ND,ND))
    Ezz = np.zeros((T,ND,ND))
    
    for t in range(T):
        Ezz[t] = V_smoother[t] + np.array([mu_smoother[t]]).transpose()@np.array([mu_smoother[t]])
    for t in range(1,T): 
        Ezz_pre[t] = V_smoother[t]@J[t-1].transpose() + np.array([mu_smoother[t]]).transpose()@np.array([mu_smoother[t-1]])
    Ezz_pre[0]=100000 #Ezz_pre[0] is not defined 
    # NOTE: Typo in the expression E[z[t]z[t-1]] in Bishop's textbook: Correctly, E[z_n z_{n-1}^T] = \hat V_n J_{n-1}^T + \hat\mu_n \hat\mu_{n-1}^T
    return Ez, Ezz_pre, Ezz


def make_double(arr):
    aux= np.copy(arr)
    aux = aux.astype(np.double)
    return aux

def lindyn_qp_Ridge(B, Ridge_designmat=0):
#For each row i of A, find Aopt_ij = argmin_{A_ij} sum_t (B2_it - sum_j B1_jt A_ij)^2  + lam*T*\sum_{j\neq i} A_ij
#with A_ij>0 and sum_j A_ij=1
#lam (>=0) is the LASSO-regularization parameter e

    ND = B.shape[-1]
    B1=B[:-1,:,:].reshape((-1,ND), order='F').T 
    B2=B[1:,:,:].reshape((-1,ND), order='F').T
    solvers.options['show_progress'] = False
    ND, NT = B2.shape
    Aopt=np.zeros((ND,ND)) 
    
    XTX=np.dot(B1,B1.T)
    # evals,evecs=np.linalg.eig(XTX)
    # absevals=np.abs(np.array(evals))
    # absevals.sort()
    # print(absevals)
    if Ridge_designmat==0:
        lam=0
    else:
        #lam =absevals[int(len(absevals)*Ridge_designmat)]
        lam = np.mean(XTX)*Ridge_designmat
    
    Lambda = np.diag(np.ones(ND)*lam)
    
    Amin=1e-6
    h = matrix(np.full(ND,-1*Amin))
    A = matrix(np.full(ND,1.0), (1,ND))
    b = matrix(1.0)
    for i in range(ND):
        Lambda_mod = Lambda.copy()
        Lambda_mod[i,i]=0
        scale = np.mean(XTX)
        P=2*matrix((XTX+Lambda_mod)/scale)
        q=-2*matrix((np.matmul(B1,B2[i]))/scale)

        G = matrix(np.diag(np.full(ND,-1.0)))# posivity
        sol=solvers.qp(P, q, G, h, A, b)
      #sol=solvers.qp(P, q) #no constraints
        Aopt[i]=np.array([sol['x'][i] for i in range(len(q))])
    return Aopt

def EM_A(Ezz_all, Ezz_pre_all, mode,ridge=0, ridge_mat=np.zeros((1,1,1))):
    solvers.options['show_progress'] = False
    
    ND = len(Ezz_all)
    A_new =np.zeros((ND,ND))
    
    qpA = matrix(np.full(ND,1.0), (1,ND))
    
    #Ridge
    XTX=np.mean(Ezz_all,axis=0)
    if ridge==0:
        lam=0
    else:
        lam = np.mean(XTX)*ridge
    
    Amin=1e-6
    h = matrix(np.full(ND,-1*Amin))
    b = matrix(1.0)
    for i in range(ND):

        G = matrix(np.diag(np.full(ND,-1.0)))        
        scale=np.mean([np.mean(Ezz_all[i]),np.mean(Ezz_pre_all[i,i])])
        
        if lam>0:
            P=matrix(make_double((Ezz_all[i]+lam*ridge_mat[i])/scale))
        else:
            P=matrix(make_double((Ezz_all[i])/scale))
            
        q=-1*matrix(make_double(Ezz_pre_all[i,i]/scale))
        
        if mode=='cstr':
            sol=solvers.qp(P, q, G, h, qpA, b)
        else:
            sol=solvers.qp(P, q) #no constraints
        A_new[i]=np.array([sol['x'][j] for j in range(len(q))])
    
    return A_new



def EM_A_L1(Ezz_all, Ezz_pre_all, mode,ridge=0, ridge_mat=np.zeros((1,1,1))):
    print('L1')
    
    solvers.options['show_progress'] = False
    
    ND = len(Ezz_all)
    A_new =np.zeros((ND,ND))
    
    qpA = matrix(np.full(ND,1.0), (1,ND))
    
    #Ridge
    XTX=np.mean(Ezz_all,axis=0)
    if ridge==0:
        lam=0
    else:
        lam = np.mean(XTX)*ridge
    
    Amin=1e-6
    h = matrix(np.full(ND,-1*Amin))
    b = matrix(1.0)
    for i in range(ND):

        G = matrix(np.diag(np.full(ND,-1.0)))        
        scale=np.mean([np.mean(Ezz_all[i]),np.mean(Ezz_pre_all[i,i])])
        
        P=matrix(make_double((Ezz_all[i])/scale))
        
        L1_i_removed=np.ones(ND)
        L1_i_removed[i]=0.0
        
        if lam>0:
            q=-1*matrix(make_double((Ezz_pre_all[i,i]-lam*L1_i_removed)/scale))
        else:
            q=-1*matrix(make_double(Ezz_pre_all[i,i]/scale))
        
        if mode=='cstr':
            sol=solvers.qp(P, q, G, h, qpA, b)
        else:
            sol=solvers.qp(P, q) #no constraints
        A_new[i]=np.array([sol['x'][j] for j in range(len(q))])
    
    return A_new





def EM_Neff(A, Ezz_all, Ezz_all_shift,Ezz_pre_all):
    ND = len(A)
    
    Ne_new =np.zeros(ND)
    for i in range(ND):
        aux0=Ezz_all_shift[i,i,i]
        aux1=-2*Ezz_pre_all[i,i]@A[i]
        aux2 = A[i]@Ezz_all[i]@A[i]
        aux_sum = np.copy(aux0 + aux1 + aux2)  
        Ne_new[i]=1/aux_sum
    
    return Ne_new


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


def Kalman_EM(counts, counts_deme, em_step_max,terminate_th=0.001, frac=0.5,noisemode=0, infer_samplenoise=True, Qprintstep=True,update_Csn='y' ,ridge=0.0,ridge_mat=np.zeros((1,1,1)),penalty_mode='L2'):
    # counts = (ND,Nlin,T) -> B = (T, Nlin, ND) 
    # counts_deme = (ND,T)
    # em_step_max: Max of EM steps
    #terminate_th: Terminate if the likelihood improvement < terminate_th
    
    
    B = (counts.copy()).transpose([2,1,0])
    T, Nlins, ND=B.shape
    
    for i in range(ND):
        for t in range(T):
            if counts_deme[i,t]>0:
                B[t,:,i]*=1.0/counts_deme[i,t]
                

    Ne_old=np.array([1000]*ND)
    
    A_LS=lindyn_qp(B,lam=0)
    
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    
    Arand = np.array( [np.random.dirichlet(([1]*ND),size=1)[0] for i in range(ND)])
    
    #A_old=np.copy(frac*A_LS+ (1.-frac)*np.ones((ND,ND))) # Start from a mixture of A_LS & homogenous 
    A_old=np.copy(frac*A_LS+ (1.-frac)*Arand) # Start from a mixture of A_LS & homogenous
    
    Csn_old=np.ones(ND)
    
    if update_Csn=='y':
        Csn_old*=1.5
    
    #print('Arand ',Arand )
    lnLH_record=[]    
    


    #print('ALS',A_opt)
    for step in range(em_step_max):
        if Qprintstep==True:
            print('step',step)
        
        #print(Csn_old)

        lnLH_all=0
        
        # Quantities writtein as E[O] in Bishop's textbook normalized by heterozygosity
        Ezz_all =np.zeros((ND,ND,ND),dtype='float32') #i,j,k component = sum_{lin,t} E[zt-1 zt-1]_jk/H_it, t =1,...,T-1. H it = f_it(1-f_it)
        Ezz_pre_all =np.zeros((ND,ND,ND),dtype='float32') # sum_{lin,t} E[zt zt-1]_jk/H_it, t =1,...,T-1.
        Ezz_all_shift =np.zeros((ND,ND,ND),dtype='float32') # sum_{lin,t} E[zt zt]_jk/H_it, t =1,...,T-1.
        
        Csn_new=np.zeros(ND,dtype='float32') # Strength of sample noise
    
       # allfreqave = np.mean(B)
        
        for lin in range(Nlins):
            
            #mu_star=np.mean(B[:,lin,:],axis=0)
            mu_star=B[0,lin,:]
                       
            V_star= np.diag([Csn_old[i]*(mu_star[i]*(1-mu_star[i]))/counts_deme[i,0] for i in range(ND)])
            
            x=B[:,lin,:].T # NDxT
            

            mu_filter, V_filter,P_filter, lnLH=Kfilter(x, mu_star, V_star, A_old, counts_deme, Ne_old,Csn_old, noisemode=noisemode)    
            mu_smoother, V_smoother, J = Ksmoother(A_old, mu_filter, V_filter,P_filter)
            
            Ez, Ezz_pre, Ezz=    Expvals(mu_smoother, V_smoother, J)# Quantities writtein as E[O] in Bishop's textbook 
 
            if noisemode==0:
                hetero =x*(1-x)  # Hetetozygosity, ND*T
                #hetero[:,0]=np.nan # The first point is not used.
            elif noisemode==1:
                hetero =np.ones((ND,T))*np.mean(x)*(1-np.mean(x))
                #hetero[:,0]=np.nan # The first point is not used.
            elif noisemode==2:
                hetero =np.zeros((ND,T))
                for i in range(ND):
                    hetero[i] = np.ones(T)*np.mean(x[i])*(1-np.mean(x[i]))
                    
            elif noisemode==3:
                hetero =x*(1-x)  # Hetetozygosity, ND*T
 
            for i in range(ND):
                if noisemode in [0,1,2]:
                    invhetero= np.diag(1./hetero[i,1:])
                elif noisemode==3:
                    invhetero= np.diag(1./hetero[i,:-1])
                
                Ezz_all[i]+=np.sum(np.tensordot(invhetero, Ezz[:-1],axes=(1,0)),axis=0)
                Ezz_all_shift[i] +=np.sum(np.tensordot(invhetero, Ezz[1:],axes=(1,0)),axis=0)
                Ezz_pre_all[i]+=np.sum(np.tensordot(invhetero, Ezz_pre[1:],axes=(1,0)),axis=0)
                
                if update_Csn=='y':
                # x,Eztr,hetero:NDxT
                    Eztr=Ez.transpose().copy()
                    Csn_new[i]+=np.sum((x[i,1:]*x[i,1:]-2*x[i,1:]*Eztr[i,1:]+ Ezz[1:,i,i])*counts_deme[i,1:]/hetero[i,1:])# The contribution from t=0 is removed because of the assumption mu_star=B[0,lin,:], which implies that Csn is a parameter for t>=1
                    #Csn_new[i]+=np.sum((x[i,0:]*x[i,0:]-2*x[i,0:]*Eztr[i,0:]+ Ezz[0:,i,i])*counts_deme[i,0:]/hetero[i,0:])
                    
                    
            lnLH_all+=lnLH
        
        Ezz_all*=1.0/(Nlins*(T-1))
        Ezz_pre_all*=1.0/(Nlins*(T-1))
        Ezz_all_shift*=1.0/(Nlins*(T-1))
        
        #M-step (Update parameters)
        if penalty_mode=='L2':
            A_new=np.copy(EM_A(Ezz_all, Ezz_pre_all,'cstr',ridge=ridge,ridge_mat=ridge_mat))
        elif penalty_mode=='L1':
            A_new=np.copy(EM_A_L1(Ezz_all, Ezz_pre_all,'cstr',ridge=ridge))
            
        DA =  np.max(np.abs(A_new-A_old))
        A_old = np.copy(A_new)
        
        Ne_new=np.copy(EM_Neff(A_old,  Ezz_all,Ezz_all_shift, Ezz_pre_all))
        DNe = np.max(np.abs(Ne_new-Ne_old)/Ne_old)
        Ne_old = np.copy(Ne_new)
        
        if update_Csn=='y':
            Csn_new*=1.0/(Nlins*(T-1))
        elif update_Csn=='n':
            for i in range(ND):
                Csn_new[i]=1.0
                
        for i in range(ND):
            if Csn_new[i]<1:
                 Csn_new[i]=1
        if infer_samplenoise==False:
            for i in range(ND):
                Csn_new[i]=1
            
        DCsn = np.max(np.abs(Csn_new-Csn_old)/Csn_old)
        Csn_old = Csn_new.copy()
        

        lnLH_record.append(lnLH_all)
        
        if step>1:
            #print("step={}, DA={}, DNe={}, DCsn={}".format(step,DA,DNe,DCsn))
            if lnLH_record[-1] - lnLH_record[-2]<0:
                
                if np.abs(lnLH_record[-1] - lnLH_record[-2])/np.abs(lnLH_record[-1]) >0.01:
                    print("LH decreses @"+str(step)+', Error?')
                    
                # np.save('err_B.npy',B)
                # np.save('err_countsdeme.npy',counts_deme)
                
            #if np.abs(lnLH_record[-1] - lnLH_record[-2])/np.abs(lnLH_record[-2]) < terminate_th:#Terminate if LH is not improved 
            
            if DA<terminate_th and DNe<0.025 and DCsn<0.025:
                print("terminate at step={}, DA={}, ratioDNe={}".format(step,np.round(DA,5),np.round(DNe,5))+', ratioDCsn='+str(np.round(DCsn,5)))
                break
        
            
    return lnLH_record, A_old, Ne_old, A_LS,Csn_old


# # flat prior on simplex. All components are non-negative
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


def MCMC(B,A_start,counts_deme, Ne_start,tmax_mcmc,dt, burnin=0.25, h=0.005,noisemode=0):
    
    Nlins = B.shape[1]
    ND=B.shape[-1]
    
    # Define the hidden-state transition noise. Should be frequency dependent
    Ne_old = np.copy(Ne_start)#np.identity(M)*1.0/Ne
    
    A_old=A_start
    LogLH_old=-1000000
    accept=0
    
    
    #Store the markov process at 100 points. Burn-in = initial 25% 
    t_out = [t  for t in range(tmax_mcmc) if t>burnin*tmax_mcmc]
    if len(t_out)>1000:
        t_out = [t  for t in range(tmax_mcmc) if t>burnin*tmax_mcmc]
        t_out = t_out[:: int(len(t_out)/1000)]
    if len(t_out)>100 and ND>10:
        t_out = [t  for t in range(tmax_mcmc) if t>burnin*tmax_mcmc]
        t_out = t_out[:: int(len(t_out)/100)]
        
    #Define the variables to store
    A_mcmc=[]
    LogLH_mcmc=[]
    Ne_mcmc=[]
        
    #run MCMC
    
    freqave = np.mean(B)
    for t_mcmc in range(tmax_mcmc):

        if t_mcmc%100==0 and t_mcmc>0:
            print(t_mcmc,' accept%', accept*100/t_mcmc)
            
        # Propose a move (proposal should be symmetric because this program is non-Hastings)
        A_new = update_A(A_old,h)
        Ne_new = np.array([i *np.exp(np.random.normal(0,h)) for i in Ne_old])#transition
 
        # Compute LH
        LogLH_new=0
        mu_star=np.array([np.mean(B[0,:,:])]*ND).T
        V_star= np.diag([np.var(B[0,:,:])]*ND)        
        
        
        for lin in range(Nlins):
            x=B[:,lin,:].T
           
            mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A_new, counts_deme, Ne_new,freqave,noisemode)  
            LogLH_new+=lnLH
            
        # acceptance_prob = min(1, np.exp(Delta)). if-else is used for numerical stability
        Delta =(LogLH_new - LogLH_old)
        if Delta>0:
            acceptance_prob=1
        else:
            acceptance_prob = np.exp(Delta)

        # update the state
        if acceptance_prob>np.random.random():
            A_old = np.copy(A_new)
            Ne_old=np.copy(Ne_new)
            LogLH_old= np.copy(LogLH_new)
            accept+=1
            
        #Store the state
        if t_mcmc in t_out:
            A_mcmc.append(np.copy(A_old))
            Ne_mcmc.append(np.copy(Ne_new))
            LogLH_mcmc.append(np.copy(LogLH_old))

    return accept/tmax_mcmc, t_out, np.array(A_mcmc), np.array(Ne_mcmc), np.array(LogLH_mcmc)



def lindyn_qp(B, lam=0):
#For each row i of A, find Aopt_ij = argmin_{A_ij} sum_t (B2_it - sum_j B1_jt A_ij)^2  + lam*T*\sum_{j\neq i} A_ij
#with A_ij>0 and sum_j A_ij=1
#lam (>=0) is the LASSO-regularization parameter e

    ND = B.shape[-1]
    B1=B[:-1,:,:].reshape((-1,ND), order='F').T 
    B2=B[1:,:,:].reshape((-1,ND), order='F').T
    solvers.options['show_progress'] = False
    ND, NT = B2.shape
    Aopt=np.zeros((ND,ND)) 
    P=2*matrix(np.dot(B1,B1.T))
    
    Amin=0.0001
    h = matrix(np.full(ND,-1*Amin))
    A = matrix(np.full(ND,1.0), (1,ND))
    b = matrix(1.0)
    for i in range(ND):
        # vec_lasso=np.array([lam*NT]*ND)
        # vec_lasso[i]=0
        #q=-2*matrix(np.matmul(B1,B2[i])-vec_lasso)
        q=-2*matrix(np.matmul(B1,B2[i]))

        G = matrix(np.diag(np.full(ND,-1.0)))
      #G[i,i]=0# Feb22: A_ii is allowed to be negative. Jul22: A_ii>0 imposed 

        sol=solvers.qp(P, q, G, h, A, b)
      #sol=solvers.qp(P, q) #no constraints
        Aopt[i]=np.array([sol['x'][i] for i in range(len(q))])
    return Aopt


def lindyn_qp_wo_CSTR(B,lam=0):
#For each row i of A, find Aopt_ij = argmin_{A_ij} sum_t (B2_it - sum_j B1_jt A_ij)^2  + lam*T*\sum_{j\neq i} A_ij
#with A_ij>0 for off-diagonal elements and sum_j A_ij=1
#lam (>=0) is the LASSO-regularization parameter 

    ND = B.shape[-1]
    B1=B[:-1,:,:].reshape((-1,ND), order='F').T 
    B2=B[1:,:,:].reshape((-1,ND), order='F').T
    
    solvers.options['show_progress'] = False
    ND, NT = B2.shape
   
    Aopt=np.zeros((ND,ND)) 
    P=2*matrix(np.dot(B1,B1.T))
    
    Amin=0.0001
    h = matrix(np.full(ND,-1*Amin))
    A = matrix(np.full(ND,0.0), (1,ND))
    b = matrix(0.0)
    for i in range(ND):
      vec_lasso=np.array([lam*NT]*ND)
      vec_lasso[i]=0
      q=-2*matrix(np.matmul(B1,B2[i])-vec_lasso)
      G = matrix(np.diag(np.full(ND,-1.0)))
      # G[i,i]=0#Jul22: A_ii>0 imposed 
      sol=solvers.qp(P, q)
        
      Aopt[i]=np.array([sol['x'][i] for i in range(len(q))])
    return Aopt



def LSWF(B):
    ND =B.shape[2]
    Aopt=np.zeros((ND,ND)) 
    Neff=np.zeros(ND) 
    solvers.options['show_progress'] = False

    for i in range(ND):
        P = np.zeros((ND,ND))
        q = np.zeros(ND)
        r=0

        num_transition=0 
        for l in range(B.shape[1]):
            for t in range(B.shape[0]-1):
                var = B[t,l,i]*(1-B[t,l,i])
                if var>0:
                    P+=np.dot(np.array([B[t,l,:]]).transpose(),np.array([B[t,l,:]]))/var
                    for j in range(ND): q[j]-= B[t+1,l,i]*B[t,l,j]/var
                    r+=0.5*B[t+1,l,i]**2/var
                    num_transition+=1
                    
        h = matrix(np.full(ND,0.0))         
        P = matrix(P)
        q = matrix(q)
        a = matrix(np.full(ND,1.0), (1,ND))
        b = matrix(1.0)
        G = matrix(np.diag(np.full(ND,-1.0)))

        sol=solvers.qp(P, q, G, h, a, b) # Min[1/2 x^T P x + q^T x]
        Aopt[i] = np.array(sol['x']).flatten()

        Neff[i] = num_transition/(Aopt[i].transpose() @ np.array(P) @ Aopt[i] + 2*np.array(q).transpose()@Aopt[i] +2* np.array(r))
        
    return Aopt, Neff


def calc_B(counts, pseudo=0):
    # B = (T, Nlin, ND) 
    # counts_deme = (ND,T)
    # em_step_max: Max of EM steps
    #terminate_th: Terminate if the likelihood improvement < terminate_th
    counts_deme = np.sum(counts,axis=1)
    B = counts.copy()
    B = B.transpose([2,1,0])  
    T, Nlins, ND=B.shape
    for t in range(T):
        for l in range(Nlins):
            for i in range(ND):
                
                B[t,l,i]+=pseudo
                B[t,l,i]=(B[t,l,i])/counts_deme[i,t]
    return B

def calc_LH(counts, A, Ne,noisemode=0):
    # B = (T, Nlin, ND) 
    # counts_deme = (ND,T)
    # em_step_max: Max of EM steps
    #terminate_th: Terminate if the likelihood improvement < terminate_th
    counts_deme = np.sum(counts,axis=1)
    B = counts.copy()
    B = B.transpose([2,1,0])  
    T, Nlins, ND=B.shape
    for t in range(T):
        for l in range(Nlins):
            for i in range(ND):
                if B[t,l,i]==0:
                    B[t,l,i]+=1
                B[t,l,i]=(B[t,l,i])/counts_deme[i,t]
    lnLH_record=[]    
    
    mu_star=np.array([np.mean(B[0,:,:])]*ND).T
    V_star= np.diag([np.var(B[0,:,:])]*ND)

    lnLHsum=0
    #print(B)
    for lin in range(Nlins):
        x=B[:,lin,:].T
        mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A, counts_deme, Ne,noisemode=noisemode) 
        lnLHsum+=lnLH
    return lnLHsum





def calc_LH_fixed_parameters(A, Ne, Csn, counts, counts_deme, infer_samplenoise=True,noisemode=2):
    # counts = (ND,Nlin,T) -> B = (T, Nlin, ND) 
    # counts_deme = (ND,T)
    # em_step_max: Max of EM steps
    #terminate_th: Terminate if the likelihood improvement < terminate_th
    
    B = (counts.copy()).transpose([2,1,0])
    T, Nlins, ND=B.shape
    
    for i in range(ND):
        for t in range(T):
            if counts_deme[i,t]>0:
                B[t,:,i]*=1.0/counts_deme[i,t]
                
    #print(B[0,0,0])
                
    if infer_samplenoise==False:
        Csn=np.ones(ND)
    
    lnLH_all=0
    
    for lin in range(Nlins):

        mu_star=B[0,lin,:]
        V_star= np.diag([Csn[i]*(mu_star[i]*(1-mu_star[i]))/counts_deme[i,0] for i in range(ND)])
        x=B[:,lin,:].T # NDxT
        mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A, counts_deme, Ne,Csn, noisemode=noisemode) 
        lnLH_all+=lnLH
    
    return lnLH_all/Nlins
        

def calc_Pfix_longtime(x,y):
    solvers.options['show_progress'] = False
    ND = x.shape[0]
    p = (x@x.T)*x.shape[0]
    v = np.sum(y@x.T,axis=0)

    h = matrix(np.full(ND,-1*0.00000001))
    b = matrix(1.0)
    q=-2*matrix(v)
    A = matrix(np.full(ND,1.0), (1,ND))
    P=2*matrix(p)
    G = matrix(np.diag(np.full(ND,-1.0)))
    sol=solvers.qp(P, q, G, h, A, b)
    pfix= np.array([sol['x'][i] for i in range(len(q))])
    return pfix
    
def calc_Pfix_unit_interval(freq,totcounts):
    
    solvers.options['show_progress'] = False
    Df = freq[:,:,1:]-freq[:,:,:-1]
    Df =Df.reshape(Df.shape[0], Df.shape[1]*Df.shape[2])
    ND =Df.shape[0]
    
    if totcounts is None:
        p=(Df@Df.T)
    else:
        #print('Accout for measurement noise ->')
        var = freq*(1-freq)
        reciprocal_totcounts = 1 / totcounts[:, np.newaxis, :]
        var *= reciprocal_totcounts
        
        var_obs = (var[:,:,1:]+var[:,:,:-1])
        var_obs=var_obs.reshape(var_obs.shape[0], var_obs.shape[1]*var_obs.shape[2])
        p=(Df@Df.T+np.diag(np.sum(var_obs,axis=1)))

    h = matrix(np.full(ND,-1*0.00000001))
    b = matrix(1.0)
    q=-2*matrix(np.array([0.]*ND))
    A = matrix(np.full(ND,1.0), (1,ND))
    P=2*matrix(p)
    G = matrix(np.diag(np.full(ND,-1.0)))
    sol=solvers.qp(P, q, G, h, A, b)
    pfix= np.array([sol['x'][i] for i in range(len(q))])
    
    return pfix
    
    
# def demo_calc_LH_fixed_parameters(A, Ne, Csn, counts, counts_deme, infer_samplenoise=True,noisemode=0):
#     # counts = (ND,Nlin,T) -> B = (T, Nlin, ND) 
#     # counts_deme = (ND,T)
#     # em_step_max: Max of EM steps
#     #terminate_th: Terminate if the likelihood improvement < terminate_th
    
#     B = (counts.copy()).transpose([2,1,0])
#     T, Nlins, ND=B.shape
    
#     for i in range(ND):
#         for t in range(T):
#             if counts_deme[i,t]>0:
#                 B[t,:,i]*=1.0/counts_deme[i,t]
                
#     if infer_samplenoise==False:
#         Csn=np.ones(ND)
    
#     lnLH_all=0
    
#     for lin in range(Nlins):

#         mu_star=B[0,lin,:]
#         V_star= np.diag([Csn[i]*(mu_star[i]*(1-mu_star[i]))/counts_deme[i,0] for i in range(ND)])
#         x=B[:,lin,:].T # NDxT
#         mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A, counts_deme, Ne,Csn, noisemode=noisemode) 
#         lnLH_all+=lnLH
    
#     return lnLH_all
        