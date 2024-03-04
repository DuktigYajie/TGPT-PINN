# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
#from torch import linspace
import os
import time

from R_data import create_residual_data, create_IC_data, create_BC_data, exact_u
from R_Plotting import R_plot, loss_plot

# Full PINN
from R_PINN_wav import NN
from R_PINN_train import pinn_train

# Transport equation GPT-PINN
from R_TGPT_activation import P
from R_TGPT_PINN import GPT
from R_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")

# Training Parameter Set
rho_training = np.linspace(1,10,31)
#rho_test = np.linspace(1.15,9.85,30)
rho_pinn_train = 1.0

# Domain and Data
Xi, Xf         =  0.0, 2*np.pi
Ti, Tf         =  0.0, 1.0
N_train, N_test, N_simple =  101, 100, 10000
IC_pts, IC_simple = 101, 101
BC_pts =  101

train_data = create_residual_data(Xi, Xf, Ti, (rho_training[-1]/rho_training[0])*Tf, N_train, N_test, N_simple)
xt_train      = train_data[0].to(device)
f_train       = train_data[1].to(device)

residual_data = create_residual_data(Xi, Xf, Ti, Tf, N_train, N_test, N_simple)
xt_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xt_test       = residual_data[2].to(device)

IC_data = create_IC_data(Xi, Xf, Ti, Tf, IC_pts, IC_simple)
IC_xt     = IC_data[0].to(device)
IC_u      = IC_data[1].to(device)

BC_data = create_BC_data(Xi, Xf, Ti, Tf, BC_pts)
BC1       = BC_data[0].to(device)
BC2       = BC_data[1].to(device)

# PINN Setting
lr_pinn     = 0.001
epochs_pinn = 120000
layers_pinn = np.array([2, 20,20,20,1])
tol_pinn    = 1e-6

# TGPT Setting
lr_tgpt          = 0.005
epochs_tgpt      = 10000
tol_tgpt         = 1e-5
layers_gpt = np.array([2, 1, 1])
P_list = np.ones(1, dtype=object)

###############################################################################
################################ Training Loop ################################
###############################################################################

########################### Full PINN Training ############################
print(f"Begin Full PINN Training: rho = {rho_pinn_train}")
pinn_train_time_1 = time.perf_counter()
PINN = NN(layers_pinn, rho_pinn_train).to(device)

pinn_losses = pinn_train(PINN, rho_pinn_train, xt_train, IC_xt, IC_u, BC1, BC2,f_train, epochs_pinn, lr_pinn, tol_pinn)

pinn_train_time_2 = time.perf_counter()
print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
    
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
P_list[0] = P(layers_pinn, w1, w2, w3, w4, b1, b2, b3, b4, a1, a2).to(device)

############################ TGPT-PINN Training ############################    
rho_loss = []
gpt_train_time_1 = time.perf_counter()
for rho in rho_training:
    c_initial  = torch.tensor([1.0],requires_grad=True)[:,None]

    TGPT_PINN = GPT(layers_gpt, rho, P_list[0:1], c_initial, f_hat, IC_u, xt_resid, IC_xt, BC1, BC2).to(device)
    
    tgpt_losses = gpt_train(TGPT_PINN, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_tgpt, lr_tgpt, tol_tgpt,testing=True)

    rho_loss.append(tgpt_losses[0].item())
    
    #R_plot(xt_test, TGPT_PINN.forward(xt_test), title=fr"TGPT_PINN Solution $\rho={round(rho,3)}$")
    #R_plot(xt_test, abs(TGPT_PINN.forward(xt_test)-exact_u(xt_test,rho).reshape(xt_test.shape[0],1)),title=fr"TGPT_PINN Solution error $\rho={round(rho,3)}$")
    #loss_plot(tgpt_losses[2], tgpt_losses[1], title=fr"TGPT_PINN Losses $\rho={round(rho,3)}$")
    #rMAE = max(sum(abs(TGPT_PINN.forward(xt_test)-exact_u(xt_test,rho)))/sum(abs(exact_u(xt_test,rho))))
    #rRMSE = max(sum((TGPT_PINN.forward(xt_test)-exact_u(xt_test,rho))**2)/sum((exact_u(xt_test,rho))**2))
    #print(f"TGPT-PINN at {rho} with the rMAE = {rMAE} and rRMSE = {rRMSE}")

gpt_train_time_2 = time.perf_counter()
print(f"\nGPT Training Time: {(gpt_train_time_2-gpt_train_time_1)/3600} Hours")

largest_loss = max(rho_loss)
largest_loss_list =rho_loss.index(largest_loss) 

print(f"Largest Loss (Using 1 Neurons): {largest_loss} at {rho_training[int(largest_loss_list)]}")


########################## TGPT-PINN Testing ######################################
rho = 3.25
c_initial  = torch.tensor([1.0],requires_grad=True)[:,None]

TGPT_PINN = GPT(layers_gpt, rho, P_list[0:1], c_initial, f_hat, IC_u, xt_resid, IC_xt, BC1, BC2).to(device)
tgpt_losses = gpt_train(TGPT_PINN, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_tgpt, lr_tgpt, tol_tgpt,testing=True)

R_plot(xt_test, exact_u(xt_test,rho), title=fr"$\rho={round(rho,3)}$")
R_plot(xt_test, TGPT_PINN.forward(xt_test), title=fr"$\rho={round(rho,3)}$")
R_plot(xt_test, abs(TGPT_PINN.forward(xt_test)-exact_u(xt_test,rho).reshape(xt_test.shape[0],1)),title=fr"$\rho={round(rho,3)}$")
loss_plot(tgpt_losses[2], tgpt_losses[1], title=fr"$\rho={round(rho,3)}$")

