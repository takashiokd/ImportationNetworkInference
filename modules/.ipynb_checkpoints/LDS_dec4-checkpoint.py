#EM algorithm Kalman
from numpy import linalg as LA

import numpy as np
from numpy.linalg import inv as minv
from cvxopt import matrix, solvers
import random


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



# def calc_LH_EM(B, A, Ne, counts_deme,noisemode):
#     # B = (T, Nlin, ND) 
#     # counts_deme = (ND,T)
#     # em_step_max: Max of EM steps
#     #terminate_th: Terminate if the likelihood improvement < terminate_th
    
    
#     T, Nlins, ND=B.shape
#     freqave = np.mean(B)
#     mu_star=np.array([np.mean(B[0,:,:])]*ND).T
#     V_star= np.diag([np.var(B[0,:,:])]*ND)


#     lnLH_all=0
#     mu_filter_all=[]
    
#     for lin in range(Nlins):
#         x=B[:,lin,:].T
#         mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A, counts_deme, Ne,freqave,noisemode=0)        
#         lnLH_all+=lnLH
#         mu_filter_all.append(mu_filter)
        
#     return np.array(mu_filter_all), lnLH_all


def filter_initial(x0, mu_pre, V_pre,  Sigma):
    M = len(Sigma)
    I = np.identity(M)
    
    # initial step
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
    
    K_n = P_old@minv( (P_old) + Sigma)
    mu_n = A@ mu_old + K_n @(x_n-A@mu_old)
    V_n =(I-K_n)@P_old 

    normal_mean = A@mu_old
    normal_cov =  P_old + Sigma
    lnc_n = logGauss(x_n, normal_mean, normal_cov)

    return mu_n, V_n, P_old, lnc_n 

    
def Kfilter(x, mu_star, V_star, A,  counts_deme, Ne, freqave,noisemode):
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
    
   
    Sigma = np.diag([ mu_star[i]*(1-mu_star[i])/np.max([counts_deme[i,0],1])  for i in range(ND)])
 
    mu, V, lnc_0=filter_initial(x[:,0], mu_star, V_star, Sigma)
   
    lnLH+=lnc_0
    
    mu_filter.append(mu)
    V_filter.append(V)
    
   
    for t in range(1,len(x[0])):
        Amu=A@mu
        
        if noisemode==0:
            Sigma = np.diag([ freqave*(1-freqave)/np.max([counts_deme[i,t],1])  for i in range(ND)])
            Gamma = np.diag([ freqave*(1-freqave)/Ne[i] for i in range(ND)])
        elif noisemode==1:
            Sigma = np.diag([ Amu[i]*(1-Amu[i])/np.max([counts_deme[i,t],1])  for i in range(ND)])
            Gamma = np.diag([ Amu[i]*(1-Amu[i])/Ne[i] for i in range(ND)])
            

        mu, V, P, lnc_n= filter_later(x[:,t], mu, V, A,  Sigma,Gamma)
        lnLH+=lnc_n
        
        mu_filter.append(mu)
        V_filter.append(V)
        P_filter.append(P)
     
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


def EM_A(Ezz_all, Ezz_pre_all, mode):
    solvers.options['show_progress'] = False
    
    ND = len(Ezz_all)
    A_new =np.zeros((ND,ND))
    
    qpA = matrix(np.full(ND,1.0), (1,ND))# This A is not the coupling matrix
    h = matrix(np.full(ND,0.0))
    b = matrix(1.0)
    for i in range(ND):
        G = matrix(np.diag(np.full(ND,-1.0)))
        P=matrix(np.copy(Ezz_all))
        q=-1*matrix(np.copy(Ezz_pre_all[i]))
        if mode=='cstr':
            sol=solvers.qp(P, q, G, h, qpA, b)
        else:
            sol=solvers.qp(P, q) #no constraints
        A_new[i]=np.array([sol['x'][i] for i in range(len(q))])
    
    return A_new

def EM_Neff(A, Ezz_all, Ezz_all_shift,Ezz_pre_all):
    ND = len(A)
    aux0=Ezz_all_shift
    aux1=-Ezz_pre_all@A.transpose()-A@Ezz_pre_all.transpose()
    aux2 = A@Ezz_all@A.transpose()
    aux_sum = np.copy(aux0 + aux1 + aux2)
  
    Ne_new = np.copy([(1/aux_sum[i,i]) for i in range(ND)])
    
    return Ne_new

def Kalman_EM(B, counts_deme, em_step_max,terminate_th, frac=0.5,noisemode=0):
    print('dec4')
    # B = (T, Nlin, ND) 
    # counts_deme = (ND,T)
    # em_step_max: Max of EM steps
    #terminate_th: Terminate if the likelihood improvement < terminate_th
    
    
    T, Nlins, ND=B.shape
    Ne_old=np.array([1000]*ND)
    
    A_LS=lindyn_qp(B,lam=0)

    A_old=np.copy(frac*A_LS+ (1.-frac)*np.ones((ND,ND))) # Start from a mixture of A_LS & homogenous 
    
    lnLH_record=[]    
    
    mu_star=np.array([np.mean(B[0,:,:])]*ND).T
    V_star= np.diag([np.var(B[0,:,:])]*ND)
    
    

    lnLH_opt = -10000000
    A_opt=np.copy(A_old)
    Ne_opt=np.copy(Ne_old)    
    for step in range(em_step_max):

        lnLH_all=0

        Ezz_all =np.zeros((ND,ND)) # E[zt-1 zt-1], t =1,...,T-1
        Ezz_pre_all =np.zeros((ND,ND))
        Ezz_all_shift =np.zeros((ND,ND)) # E[zt zt], t =1,...,T-1

        for lin in range(Nlins):
            x=B[:,lin,:].T
            freqave = np.mean(x) # Variance is assumed to be constant over time and demes for each lineage
            if freqave==0:
                print('error: freqave=0')
            var_WF = freqave*(1-freqave)
            
            mu_filter, V_filter,P_filter,  lnLH=Kfilter(x, mu_star, V_star, A_old, counts_deme, Ne_old,freqave,noisemode=0)    
            mu_smoother, V_smoother, J = Ksmoother(A_old, mu_filter, V_filter,P_filter)
            
            Ez, Ezz_pre, Ezz=    Expvals(mu_smoother, V_smoother, J)    
            Ezz_all+=np.sum(Ezz[:-1],axis=0)*1/var_WF 
            Ezz_all_shift +=np.sum(Ezz[1:],axis=0)*1/var_WF 
            Ezz_pre_all+=np.sum(Ezz_pre[1:],axis=0)*1/var_WF 
            lnLH_all+=lnLH
            
        Ezz_all*=1.0/(Nlins*(T-1))
        Ezz_pre_all*=1.0/(Nlins*(T-1))
        Ezz_all_shift*=1.0/(Nlins*(T-1))
        
        #M-step (Update parameters)
        A_new=np.copy(EM_A(Ezz_all, Ezz_pre_all,'cstr'))
        A_old = np.copy(A_new)
        Ne_new=np.copy(EM_Neff(A_old,  Ezz_all,Ezz_all_shift, Ezz_pre_all))
        Ne_old = np.copy(Ne_new)

        lnLH_record.append(lnLH_all)
        
        if step>1:
            if lnLH_record[-1] - lnLH_record[-2]<0:
                print("LH decreses")
                if np.abs(lnLH_record[-1] - lnLH_record[-2])/np.abs(lnLH_record[-1]) <0.001:
                    print("(due to numerical error")
                else:
                    print('Maybe bug?')
                    
                np.save('err_B.npy',B)
                np.save('err_countsdeme.npy',counts_deme)
                
            if (lnLH_record[-1] - lnLH_record[-2])/np.abs(lnLH_record[-2]) < terminate_th:#Terminate if LH is not improved 
                print("terminate at step ", step)
                break
            
        
    return lnLH_record, A_old, Ne_old







# # Old version, where offdiagonal components have flat prior on [0,1].
# # This means that diagonal compoenents can be negative
# def dx_randomwalk(x,h):
#     dx=np.random.normal(0,h)
#     if x+dx<0:
#         #x=-x-dx
#         dx=-2*x-dx
#     elif x+dx>1:
#         #x=2-x-dx
#         dx=2-2*x-dx
#     return dx
# def old_update_A(A,h):
#     res=np.copy(A)
#     n=len(A)
#     for i in range(n):
#         for j in range(n):
#             if j!=i:
#                 da=dx_randomwalk(A[i,j],h)
#                 res[i,j]+=da
#                 res[i,i]-=da
#     return res





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
#lam (>=0) is the LASSO-regularization parameter 
# Feb22: A_ii is allowed to be negative

    ND = B.shape[-1]
    B1=B[:-1,:,:].reshape((-1,ND), order='F').T 
    B2=B[1:,:,:].reshape((-1,ND), order='F').T
    solvers.options['show_progress'] = False
    ND, NT = B2.shape
    Aopt=np.zeros((ND,ND)) 
    P=2*matrix(np.dot(B1,B1.T))
    
    h = matrix(np.full(ND,0.0))
    A = matrix(np.full(ND,1.0), (1,ND))
    b = matrix(1.0)
    for i in range(ND):
      vec_lasso=np.array([lam*NT]*ND)
      vec_lasso[i]=0
      q=-2*matrix(np.matmul(B1,B2[i])-vec_lasso)

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

    h = matrix(np.full(ND,0.0))
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


# def LSWF(B):
#     ND = B.shape[2]
#     X=B[:-1].reshape((-1,ND), order='F').T
#     Y=B[1:].reshape((-1,ND), order='F').T
#     Xbar=1/(X*(1-X))
#     Ytilde=Y*Xbar
#     Ctau=Ytilde @ X.T
#     A=[]
#     for i in range(X.shape[0]):
#         Xii=np.array([Xbar[i] for j in range(X.shape[0])]) 
#         Czero=X @ (X * Xii).T
#         A.append(np.linalg.pinv(Czero) @ Ctau[i])
#     A=np.array(A)
#     Delta=Y-A@X
#     pop_sizes=[]
#     for i in range(X.shape[0]):
#         pop_sizes.append(X.shape[1]/(Delta[i] @ (Delta[i] * Xbar[i])))
     
#     return A, np.array(pop_sizes)



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