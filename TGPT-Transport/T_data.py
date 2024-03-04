import numpy as np
import torch
import torch.autograd as autograd
from torch import linspace, meshgrid, hstack, zeros, sin, pi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from choice_widthindex import random_point_in_diagonal_band, random_point_in_initial

def exact_u(xt,nu):
    condition=abs((xt[:,0]-nu*xt[:,1]+1) % 2 -1 )<0.5
    u=torch.where(condition,torch.tensor(0.0), torch.tensor(1.0)).to(torch.float32)
    return u

def initial_u(x, t=0):
    u = torch.where(abs(x) < 0.5, torch.tensor(0.0), torch.tensor(1.0)).to(torch.float32)
    return u


def create_IC_data(Xi, Xf, Ti, Tf, IC_pts, IC_simple):
    ##########################################################   
    x_IC = linspace(Xi, Xf, IC_pts)
    t_IC = linspace(Ti, Ti, IC_pts)
    X_IC, T_IC = meshgrid(x_IC, t_IC, indexing='ij')
    id_ic = np.zeros(IC_simple,dtype=int)
    for i in range(0, IC_simple):
        new_id = random_point_in_initial(IC_pts, 2000)
        if (new_id in id_ic[0:i]):
            new_id = random_point_in_initial(IC_pts, 2000)    
        else:
            id_ic[i] = new_id
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

def create_residual_data(nu, Xi, Xf, Ti, Tf, N_train, N_test, N_simple):
    ##########################################################
    x_resid = linspace(Xi, Xf, N_train)
    t_resid = linspace(Ti, Tf, N_train)
    
    XX_resid, TT_resid = meshgrid((x_resid, t_resid), indexing='ij')
    
    X_resid = XX_resid.transpose(1,0).flatten()[:,None]
    T_resid = TT_resid.transpose(1,0).flatten()[:,None]
    ##########################################################    
    id_f =np.zeros(N_simple,dtype=int)   
    for i in range(0, int(np.sqrt(N_simple))):
        id_f[i] = random_point_in_initial(N_train, 40)           
    for i in range(int(np.sqrt(N_simple)), N_simple):        
        id_f[i] = random_point_in_diagonal_band(nu, N_train, 40)
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
    f_hat_test = zeros((xt_test.shape[0], 1))
    ##########################################################
    return (xt_resid_rd, f_hat_train, xt_test, f_hat_test)
