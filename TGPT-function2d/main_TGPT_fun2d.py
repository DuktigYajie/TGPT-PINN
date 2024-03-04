# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
#import os
import time
from functools import partial

# GPT-NN
from fun2d_TGPT_activation import P
from fun2d_TGPT_PINN import GPT
from fun2d_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device: {device}')
if torch.cuda.is_available():
    print(f'Current Device Name: {torch.cuda.get_device_name()}')

def exact_u(nu, xy):    
    ##########################################################
    x=xy[:,0]
    y=xy[:,1]
    nu1 = nu[0]
    nu2 = nu[1]
    u_xy = 1.0/torch.sqrt(torch.add(torch.pow(torch.sub(x,nu1),2),torch.pow(torch.sub(y,nu2),2)))
    return u_xy

def create_xy_data(Xi, Xf, Yi, Yf, Nx, Ny):
    ##########################################################
    train_x = torch.linspace(Xi, Xf, Nx)
    train_y = torch.linspace(Yi, Yf, Ny)
    
    yy, xx = torch.meshgrid((train_x, train_y), indexing='ij')    
    X = xx.transpose(1,0).flatten()[:,None]
    Y = yy.transpose(1,0).flatten()[:,None]
    
    u_xy   = torch.hstack((X, Y))
    return X, Y, u_xy

# Domain
Xi, Xf              = 0.0, 1.0
Yi, Yf              = 0.0, 1.0
Nx_train, Ny_train     =  101, 101

uxy_data   =   create_xy_data(Xi, Xf, Yi, Yf, Nx_train, Ny_train)
train_x    =   uxy_data[0].to(device)
train_y    =   uxy_data[1].to(device)
train_xy    =   uxy_data[2].to(device)

# Training Parameter Set
nu_training = np.linspace(-1.0,-0.01, 21)
nuxy_training=[]
for i in range(nu_training.shape[0]):
    for j in range(nu_training.shape[0]):
        nuxy_training.append([nu_training[i],nu_training[j]])
nuxy_training = np.array(nuxy_training)

# Activation functions
P_nu = [-1, -1]
P_list = np.ones(1, dtype=object)
P_func_nu = partial(exact_u, P_nu)
P_list[0] = P(P_func_nu).to(device)

# TGPT-PINN Setting
lr_tgpt          = 0.025
epochs_tgpt      = 2000
tol_tgpt         = 1e-11

layers_gpt = np.array([1, 1, 1, 1])
initial_c  = torch.tensor([1.0],requires_grad=True)[:,None]
initial_s  = torch.tensor([0.0],requires_grad=True)[:,None]
############################ TGPT-PINN Training ############################   
nu_loss = []
        
tgpt_train_time_1 = time.perf_counter()

for nu in nuxy_training:

    u_exact  =  exact_u(nu, train_xy).reshape(train_xy.shape[0],1)

    TGPT_NN   = GPT(layers_gpt, nu, P_list[0:1], initial_c, initial_s,train_xy, u_exact).to(device)

    tgpt_losses = gpt_train(TGPT_NN, layers_gpt, nu, P_list[0:1], train_xy, u_exact, epochs_tgpt, lr_tgpt, tol_tgpt)
    
    nu_loss.append(tgpt_losses[0]) 

tgpt_train_time_2 = time.perf_counter()
print("\nTGPT-PINN Training Completed")
print(f"TGPT Training Time ({i+1} Neurons): {(tgpt_train_time_2-tgpt_train_time_1)/3600} Hours")
    
largest_loss = max(nu_loss)
largest_loss_list=nu_loss.index(largest_loss)
print(f"Largest Loss (Using 1 Neurons): {largest_loss} at {nuxy_training[int(largest_loss_list)]}")