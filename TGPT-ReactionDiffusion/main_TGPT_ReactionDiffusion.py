# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
#from torch import linspace
import os
import time

from RD_data import create_residual_data, create_IC_data, create_BC_data
from RD_Plotting import RD_plot,loss_plot

# Full PINN
from RD_PINN_wav import NN
from RD_PINN_train import pinn_train

# Reaction-Diffusion equation TGPT-PINN
from RD_TGPT_activation import P
from RD_TGPT_PINN import GPT
from RD_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")

# Domain and Data
Xi, Xf         =  0.0, 2*np.pi
Ti, Tf         =  0.0, 1.0
N_train, N_test, N_simple =  101, 88, 10000
IC_pts, IC_simple = 101,101
BC_pts =  101

residual_data = create_residual_data(Xi, Xf, Ti, Tf, N_train, N_test, N_simple)
xt_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xt_test       = residual_data[2].to(device)

IC_data = create_IC_data(Xi, Xf, Ti, Tf, IC_pts, IC_simple)
IC_xt     = IC_data[0].to(device)
IC_u      = IC_data[1].to(device)

BC_data = create_BC_data(Xi, Xf, Ti, Tf, BC_pts)
BC1     = BC_data[0].to(device)
BC2       = BC_data[1].to(device)

# Training Parameter Set
nu_training = np.linspace(5, 1, 11)
rho_training  = np.linspace(5, 1, 11)

db_training = []
for i in range(nu_training.shape[0]):
    for j in range(rho_training.shape[0]):
        db_training.append([nu_training[i],rho_training[j]])
db_training = np.array(db_training)

###############################################################################
#################################### Setup ####################################
###############################################################################
number_of_neurons = 10
db_neurons    = [1 for i in range(number_of_neurons)]
db_neurons[0] =  [1.0, 5.0]
loss_list         = np.ones(number_of_neurons) 
P_list = np.ones(number_of_neurons, dtype=object)
print(f"Expected Final GPT-PINN Depth: {[2,number_of_neurons,1]}\n")

# PINN Setting
lr_pinn     = 0.001
epochs_pinn = 120000
layers_pinn = np.array([2, 40,40,40,1])
tol_pinn    = 1e-6

# TGPT-PINN Setting
lr_tgpt          = 0.005
epochs_tgpt      = 10000
tol_tgpt         = 1e-5

total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(0, number_of_neurons):
    print("******************************************************************")
    ########################### Full PINN Training ############################
    db_pinn_train = db_neurons[i]
    nu_pinn_train, rho_pinn_train = db_pinn_train[0], db_pinn_train[1] 
    #spilt_exact_u=torch.from_numpy(exact_u(Xi,Xf,Ti,Tf,N_test,nu_pinn_train,rho_pinn_train).reshape(xt_test.shape[0],1)).to(device)
    print(f"Begin Full PINN Training: nu ={nu_pinn_train}, rho = {rho_pinn_train}")

    pinn_train_time_1 = time.perf_counter()
    PINN = NN(layers_pinn, nu_pinn_train, rho_pinn_train).to(device)

    pinn_losses = pinn_train(PINN, xt_resid, IC_xt, IC_u, BC1, BC2,f_hat, epochs_pinn, lr_pinn, tol_pinn)

    pinn_train_time_2 = time.perf_counter()
    print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    
    #rMAE = max(sum(abs(PINN(xt_test)-spilt_exact_u))/sum(abs(spilt_exact_u)))
    #rRMSE = torch.sqrt(sum((PINN(xt_test)-spilt_exact_u)**2)/sum((spilt_exact_u)**2)).item()
    #print(f"TGPT-PINN at {db_pinn_train} with the rMAE = {rMAE} and rRMSE = {rRMSE}")

    w1 = PINN.linears[0].weight.detach().cpu()
    w2 = PINN.linears[1].weight.detach().cpu()
    w3 = PINN.linears[2].weight.detach().cpu()
    w4 = PINN.linears[3].weight.detach().cpu()
    
    b1 = PINN.linears[0].bias.detach().cpu()
    b2 = PINN.linears[1].bias.detach().cpu()
    b3 = PINN.linears[2].bias.detach().cpu()
    b4 = PINN.linears[3].bias.detach().cpu()

    a1 = PINN.activation.w1.detach().cpu()
    a2 = PINN.activation.w2.detach().cpu()
        
    # Add new activation functions
    P_list[i] = P(layers_pinn, w1, w2, w3, w4, b1, b2, b3, b4, a1, a2).to(device)

    print(f"\nCurrent TGPT-PINN Depth: [2,{i+1},1]")

    ############################ GPT-PINN Training ############################    
    layers_tgpt      = np.array([2, i+1, 1])
    c_initial  = torch.full((1,i+1), 1/(i+1))

    largest_case = 0
    largest_loss = 0

    tgpt_train_time_1 = time.perf_counter()
    for db in db_training:
        nu  = db[0]
        rho = db[1]

        TGPT_PINN = GPT(layers_tgpt, nu, rho, P_list[0:i+1], c_initial, f_hat, IC_u, xt_resid, IC_xt, BC1, BC2).to(device)
    
        tgpt_losses = gpt_train(TGPT_PINN,nu, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_tgpt, lr_tgpt, largest_loss, largest_case)
    
        largest_loss = tgpt_losses[0]
        largest_case = tgpt_losses[1]

    tgpt_train_time_2 = time.perf_counter()
    print("TGPT-PINN Training Completed")
    print(f"\nTGPT Training Time: {(tgpt_train_time_2-tgpt_train_time_1)/3600} Hours")

    loss_list[i] = largest_loss
    
    if (i+1 < number_of_neurons):
        db_neurons[i+1] = largest_case

    print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
    print(f"Parameter Case: {largest_case}")

total_train_time_2 = time.perf_counter()

###############################################################################
# Results of largest loss, parameters chosen, and times may vary based on
# the initialization of full PINN and the final loss of the full PINN
print("******************************************************************")
print("*** Full PINN and TGPT-PINN Training Complete ***")
print(f"Total Training Time: {(total_train_time_2-total_train_time_1)/3600} Hours\n")
print(f"Final TGPT-PINN Depth: {[2,len(P_list),1]}")
print(f"\nActivation Function Parameters: \n{db_neurons}\n")

########################## TGPT-PINN Testing ##########################
nu  = 1.5
rho = 1.5

TGPT_PINN = GPT(layers_tgpt, nu, rho, P_list[0:i+1], c_initial, f_hat, IC_u, xt_resid, IC_xt, BC1, BC2).to(device)
test_losses = gpt_train(TGPT_PINN,nu, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_tgpt, lr_tgpt,testing=True)

#spilt_exact_u=torch.from_numpy(exact_u(Xi,Xf,Ti,Tf,N_test,nu,rho).reshape(xt_test.shape[0],1)).to(device)
RD_plot(xt_test, TGPT_PINN.forward(xt_test), title=fr"TGPT_PINN Solution at ${db}$")
loss_plot(test_losses[2], test_losses[1], title=fr"TGPT_PINN Losses ${db}$")
#rMAE = max(sum(abs(TGPT_PINN.forward(xt_test)-spilt_exact_u))/sum(abs(spilt_exact_u)))
#rRMSE = torch.sqrt(sum((TGPT_PINN.forward(xt_test)-spilt_exact_u)**2)/sum((spilt_exact_u)**2))
#print(f"TGPT-PINN at {db} with the rMAE = {rMAE} and rRMSE = {rRMSE.item()}")
