# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
#import os
import time
from functools import partial

# GPT-NN
from kink_TGPT_activation import P
from kink_TGPT_PINN import GPT
from kink_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current Device: {device}')
if torch.cuda.is_available():
    print(f'Current Device Name: {torch.cuda.get_device_name()}')

def exact_u(nu, x):
    h_x=torch.sub(torch.div(x,0.4+nu),1).to(device)
    psi_x=torch.exp(torch.reciprocal(torch.sub(torch.pow(h_x,2),1))).to(device)
    u_x = torch.mul(torch.mul(torch.where(h_x>-1, torch.ones(x.shape[0],1).to(device), torch.zeros([x.shape[0],1]).to(device)),torch.where(h_x<-1/2,torch.ones(x.shape[0],1).to(device), torch.zeros([x.shape[0],1]).to(device))), psi_x)
    u_x = torch.where(torch.isnan(u_x), torch.full_like(u_x, 0), u_x)

    return u_x

# Domain
Xi, Xf              = -1.0, 1.0
N_train, N_test     =  100, 100
train_x   =   torch.linspace(Xi, Xf, N_train)[:,None]

# Training Parameter Set
number_of_parameters = 11
nu_training = np.linspace(0, 1, number_of_parameters)
nu_plot = nu_training[0:nu_training.shape[0]:5]

number_of_neurons = 10
nu_neurons    = torch.zeros(number_of_neurons).to(device) # Neuron parameters
nu_neurons[0] = 1.0

largest_loss           = np.zeros(number_of_neurons)
largest_loss_list      = np.ones(number_of_neurons)
P_list = np.ones(number_of_neurons, dtype=object)
print(f"Expected Final GPT-PINN Depth: {[1,number_of_neurons,1]}\n")


# TGPT-PINN Setting
lr_tgpt          = 0.01
epochs_tgpt      = 5000
x_tol_tgpt       = 1e-14
u_tol_tgpt       = 1e-6

total_train_time_1 = time.perf_counter()
# ############################### Training Loop ################################
# ##############################################################################
for i in range(0, number_of_neurons):

    # Add new activation functions
    P_nu = nu_neurons[i]
    P_func_nu = partial(exact_u, P_nu)
    P_list[i] = P(P_func_nu).to(device)

    ############################ TGPT-PINN Training ############################    
    # Finding The Next Neuron
    nu_x_loss = []
    nu_u_loss = []

    layers_gpt      = np.array([1, i+1, 1])
    c_initial  = torch.full((1,i+1), 1/(i+1))
    for nu in nu_training:
        u_exact  =  exact_u(nu, train_x)

        TGPT_NN   = GPT(layers_gpt, nu, P_list[0:i+1],nu_neurons[0:i+1], c_initial, train_x, u_exact).to(device)

        tgpt_losses = gpt_train(TGPT_NN, layers_gpt, nu, epochs_tgpt, lr_tgpt,x_tol_tgpt,u_tol_tgpt)

        nu_u_loss.append(tgpt_losses[1])

    print("\nTGPT-PINN Training Completed")
    
    largest_loss[i] = max(nu_u_loss)
    largest_loss_list[i]=nu_u_loss.index(largest_loss[i])
    print(f"Largest Loss (Using 1 Neurons): {largest_loss[i]} at {nu_training[int(largest_loss_list[i])]}")
    
    if (i+1 < number_of_neurons):
        nu_neurons[i+1] = nu_training[int(largest_loss_list[i])]   
        print(f"Next parameter Case: {nu_neurons[i+1]}")
        
total_train_time_2 = time.perf_counter()
print(f"Find parameters:{nu_neurons}")




