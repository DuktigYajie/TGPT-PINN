# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
#import os
import time
from functools import partial

# TGPT-NN
from F_TGPT_activation import P
from F_TGPT_PINN import GPT
from F_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device: {device}')
if torch.cuda.is_available():
    print(f'Current Device Name: {torch.cuda.get_device_name()}')

def exact_u(nu, x):
    u_x=torch.sin(x+nu)
    #u_x=abs(x+nu)
    #u_x = torch.max(torch.sin(x+nu), torch.zeros([x.shape[0],1]).to(device))
    return u_x

# Domain
Xi, Xf    =  -np.pi, np.pi
N_train   =  201
train_x   =  torch.linspace(Xi, Xf, N_train)[:,None]

# Training Parameter Set
number_of_parameters = 11
nu_training = np.linspace(-5,5, number_of_parameters)

# TGPT-PINN Setting
lr_tgpt          = 0.01
epochs_tgpt      = 10000
tol_tgpt         = 1e-12

layers_gpt = np.array([1, 1, 1])
initial_c  = torch.tensor([1.0],requires_grad=True)[:,None]

# Activation functions
P_nu = 0.0
P_list = np.ones(1, dtype=object)
P_func_nu = partial(exact_u, P_nu)
P_list[0] = P(P_func_nu).to(device)

############################ TGPT-PINN Training ############################    
nu_loss = []
        
tgpt_train_time_1 = time.perf_counter()
for nu in nu_training:

    u_exact  =  exact_u(nu, train_x)

    TGPT_NN   = GPT(layers_gpt, nu, P_list[0:1],P_nu, initial_c,train_x, u_exact).to(device)

    tgpt_losses = gpt_train(TGPT_NN, layers_gpt, nu, P_list[0:1], train_x, epochs_tgpt, lr_tgpt, tol_tgpt)
    
    nu_loss.append(tgpt_losses[0])
    
tgpt_train_time_2 = time.perf_counter()
print("\nTGPT-PINN Training Completed")
print(f"TGPT Training Time: {(tgpt_train_time_2-tgpt_train_time_1)/3600} Hours")
    
largest_loss = max(nu_loss)
largest_loss_list=nu_loss.index(largest_loss)
print(f"Largest Loss (Using 1 Neurons): {largest_loss} at {nu_training[int(largest_loss_list)]}")
