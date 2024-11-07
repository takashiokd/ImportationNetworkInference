#Tools for stepping-stone model
from numpy import linalg as LA

import numpy as np
from numpy.linalg import inv as minv
from cvxopt import matrix, solvers
import random


def A_SS_periodic(a,p,Lx,Ly,da=0.5):
    ND = Lx*Ly
    A = np.zeros((ND,ND))

    def labeling(x,y,Lx,Ly):
        return Lx*y + x 
    
    def rewire(start, end, p, ND):
        if p==0:
            res=end
        else:
            wt = [p/(ND-1)]*ND
            wt[start]=0
            wt[end]+=1-p
            res = np.random.choice(range(ND), p=wt)
        return res
    
    def rand_rate(a,da):
        da =da*a
        return np.random.uniform(low=a-da, high=a+da)
        #return np.random.exponential(scale=a)
    
        
    for y in range(Ly):
        for x in range(Lx):
            i = labeling(x,y,Lx,Ly)
            if x==Lx-1:
                i_right =labeling(0,y,Lx,Ly)
            else:
                i_right =labeling(x+1,y,Lx,Ly)

            if x==0:
                i_left = labeling(Lx-1,y,Lx,Ly)
            else:
                i_left = labeling(x-1,y,Lx,Ly)

            if y==0:
                i_down = labeling(x,Ly-1,Lx,Ly)
            else:
                i_down = labeling(x,y-1,Lx,Ly)

            if y==Ly-1:
                i_up = labeling(x,0,Lx,Ly)
            else:
                i_up = labeling(x,y+1,Lx,Ly)
                
            
            # for iter in range(4):
            #     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
                
#             if Lx>2:
#                 if np.random.uniform()>p:
#                     A[i_right,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
            
#                 if np.random.uniform()>p:
#                     A[ i_left,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
               
#             elif Lx==2:
#                 if np.random.uniform()>p:
#                     A[i_right,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
#             elif Lx==1:
#                 ()

#             if Ly>2:
#                 if np.random.uniform()>p:
#                     A[i_up,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
            
#                 if np.random.uniform()>p:
#                     A[ i_down,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
#             elif Ly==2:
#                 if np.random.uniform()>p:
#                     A[ i_up,i]+=rand_rate(a,da)
#                 else:
#                     A[np.random.choice(range(ND)),i]+=rand_rate(a,da)
                
#             elif Ly==1:
#                 ()
                
            if Lx>2:
                A[rewire(i, i_right, p, ND),i]+=rand_rate(a,da)
                A[rewire(i, i_left, p, ND),i]+=rand_rate(a,da)
               
            elif Lx==2:
                A[rewire(i, i_right, p, ND),i]+=rand_rate(a,da)
            elif Lx==1:
                ()

            if Ly>2:
                A[rewire(i, i_up, p, ND),i]+=rand_rate(a,da)
                A[rewire(i, i_down, p, ND),i]+=rand_rate(a,da)
            elif Ly==2:
                A[rewire(i, i_up, p, ND),i]+=rand_rate(a,da)
            elif Ly==1:
                ()
    for i in range(ND):
        if A[i,i]!=0:
            print('diag')
        else:
            A[i,i] = 1-np.sum(A[i])
            if A[i,i]<0:
                print('Neg diag')
                
    return A


def A_SS(a,p,Lx,Ly,da=0.5):
    ND = Lx*Ly
    A = np.zeros((ND,ND))

    def labeling(x,y,Lx,Ly):
        return Lx*y + x 
    
    def rewire(start, end, p, ND):
        if p==0:
            res=end
        else:
            wt = [p/(ND-1)]*ND
            wt[start]=0
            wt[end]+=1-p
            res = np.random.choice(range(ND), p=wt)
        return res
    
    def rand_rate(a,da):
        da =da*a
        return np.random.uniform(low=a-da, high=a+da)
        #return np.random.exponential(scale=a)
    
        
    for y in range(Ly):
        for x in range(Lx):
            i = labeling(x,y,Lx,Ly)
            if x==Lx-1:
                i_right =labeling(0,y,Lx,Ly)
            else:
                i_right =labeling(x+1,y,Lx,Ly)

            if x==0:
                i_left = labeling(Lx-1,y,Lx,Ly)
            else:
                i_left = labeling(x-1,y,Lx,Ly)

            if y==0:
                i_down = labeling(x,Ly-1,Lx,Ly)
            else:
                i_down = labeling(x,y-1,Lx,Ly)

            if y==Ly-1:
                i_up = labeling(x,0,Lx,Ly)
            else:
                i_up = labeling(x,y+1,Lx,Ly)
                
            if Lx>2:
                if i<Lx-1:
                    A[rewire(i, i_right, p, ND),i]+=rand_rate(a,da)
                if i>0:
                    A[rewire(i, i_left, p, ND),i]+=rand_rate(a,da)
               
            elif Lx==2:
                A[rewire(i, i_right, p, ND),i]+=rand_rate(a,da)
            elif Lx==1:
                ()

            if Ly>2:
                if i<Ly-1:
                    A[rewire(i, i_up, p, ND),i]+=rand_rate(a,da)
                if i>0:
                    A[rewire(i, i_down, p, ND),i]+=rand_rate(a,da)
            elif Ly==2:
                A[rewire(i, i_up, p, ND),i]+=rand_rate(a,da)
            elif Ly==1:
                ()
                
    for i in range(ND):
        if A[i,i]!=0:
            print('diag')
        else:
            A[i,i] = 1-np.sum(A[i])
            if A[i,i]<0:
                print('Neg diag')
                
    
                
    return A



def A_uniform(a,Lx,Ly,da=0.5):
    ND = Lx*Ly
    A = np.zeros((ND,ND))

    def labeling(x,y,Lx,Ly):
        return Lx*y + x 
    
    def rewire(start, end, p, ND):
        if p==0:
            res=end
        else:
            wt = [p/(ND-1)]*ND
            wt[start]=0
            wt[end]+=1-p
            res = np.random.choice(range(ND), p=wt)
        return res
    
    def rand_rate(a,da):
        da =da*a
        return np.random.uniform(low=a-da, high=a+da)
        #return np.random.exponential(scale=a)
    
    for i in range(ND):
        for j in range(ND):
            if i!=j:
                A[j,i]+=rand_rate(a,da)
  
    for i in range(ND):
        A[i,i] = 1-np.sum(A[i])
    if np.min(A)<0:
        print('Neg')
                
    
                
    return A