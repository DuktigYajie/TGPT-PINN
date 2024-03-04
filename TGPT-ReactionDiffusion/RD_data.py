import numpy as np
import torch
import torch.autograd as autograd
from torch import linspace, meshgrid, hstack, zeros, sin, pi, ones, exp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initial_u(x, t=0):
    u = exp(-(x-pi)**2/(2*(pi/4)**2))
    return u

def exact_u(Xi,Xf,Ti,Tf,N_test,nu,rho):
    x = np.linspace(Xi, Xf, N_test, dtype=np.float32)
    dx = (Xf-Xi)/N_test
    Nt_test=100*N_test
    t = np.linspace(Ti,Tf,Nt_test, dtype=np.float32)
    dt = (Tf-Ti)/Nt_test
    u= np.zeros((N_test,Nt_test),dtype=np.float32)
    u[:,0] = np.exp(-(x-np.pi)**2/(2*(np.pi/4)**2))
    IKX_pos = 1j * np.arange(0, N_test/2+1, 1)
    IKX_neg = 1j * np.arange(-N_test/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = (IKX * IKX).astype(np.complex64)
    for n in range(Nt_test-1):
        u_half  = (u[:,n]*np.exp(rho*dt))/(u[:,n]*np.exp(rho*dt)+1-u[:,n])
        u[:,n+1] = np.fft.ifft(np.fft.fft(u_half)*np.exp((dt)*nu*IKX2)).real

    u_output = u[:, ::100].T
    return u_output

def create_IC_data(Xi, Xf, Ti, Tf, IC_pts, IC_simple):
    ##########################################################   
    x_IC = linspace(Xi, Xf, IC_pts)
    t_IC = linspace(Ti, Ti, IC_pts)
    X_IC, T_IC = meshgrid(x_IC, t_IC, indexing='ij')

    id_ic = np.random.choice(IC_pts, IC_simple, replace=False)  
    IC_x = X_IC[id_ic,0][:,None]
    IC_t = zeros(IC_x.shape[0], 1)
    IC_u = initial_u(IC_x)     
    IC   = hstack((IC_x, IC_t))
    return (IC, IC_u)  

def create_BC_data(Xi, Xf, Ti, Tf, BC_pts):
    ##########################################################
    x_BC = linspace(Xi, Xf, BC_pts)
    t_BC = linspace(Ti, Tf, BC_pts)
    X_BC, T_BC = meshgrid(x_BC, t_BC, indexing='ij')

    BC_bottom_x = X_BC[0,:][:,None] 
    BC_bottom_t = T_BC[0,:][:,None] 
    BC_bottom   = hstack((BC_bottom_x, BC_bottom_t)) 

    BC_top_x = X_BC[-1,:][:,None] 
    BC_top_t = T_BC[-1,:][:,None] 
    BC_top   = hstack((BC_top_x, BC_top_t))
    return BC_bottom, BC_top

def create_residual_data(Xi, Xf, Ti, Tf, N_train, N_test, N_simple):
    ##########################################################
    x_resid = linspace(Xi, Xf, N_train)
    t_resid = linspace(Ti, Tf, N_train)
    
    XX_resid, TT_resid = meshgrid((x_resid, t_resid), indexing='ij')
    
    X_resid = XX_resid.transpose(1,0).flatten()[:,None]
    T_resid = TT_resid.transpose(1,0).flatten()[:,None]
    
    xt_resid    = hstack((X_resid, T_resid))

    id_f =np.random.choice(N_train**2, N_simple, replace=False)  
    x_int = X_resid[:, 0][id_f, None]                                        
    t_int = T_resid[:, 0][id_f, None]                                        
    xt_resid_rd = hstack((x_int, t_int))
    f_hat_train = zeros((xt_resid_rd.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test)
    t_test = linspace(Ti, Tf, N_test)
    
    XX_test, TT_test = meshgrid((x_test, t_test), indexing='ij')
    
    X_test = XX_test.transpose(1,0).flatten()[:,None]
    T_test = TT_test.transpose(1,0).flatten()[:,None]
    
    xt_test    = hstack((X_test, T_test))
    ##########################################################
    return (xt_resid_rd, f_hat_train, xt_test)
