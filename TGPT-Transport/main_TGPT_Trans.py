# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
#from torch import linspace
import os
import time

from T_data import create_residual_data, create_IC_data, create_BC_data, exact_u
from T_Plotting import Trans_plot, loss_plot 

# Full PINN
from T_PINN import NN
from T_PINN_train import pinn_train

# Transport equation TGPT-PINN
from T_TGPT_activation import P
from T_TGPT_PINN import GPT
from T_TGPT_train import gpt_train

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")
if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name()}")


nu_pinn_train = 0.0

lr_pinn     = 0.001
epochs_pinn = 400000
layers_pinn = np.array([2, 20,20,20,1])
tol         = 1e-6

# Domain and Data
Xi, Xf         = -1.0, 1.0
Ti, Tf         =  0.0, 2.0
N_train, N_test, N_simple =  201, 100, 10000
IC_pts, IC_simple = 10001, 1000
BC_pts =  1001

residual_data = create_residual_data(nu_pinn_train, Xi, Xf, Ti, Tf, N_train, N_test, N_simple)
xt_resid      = residual_data[0].to(device)
f_hat         = residual_data[1].to(device)
xt_test       = residual_data[2].to(device)
#plt.scatter(xt_resid[:,1].detach().cpu(),xt_resid[:,0].detach().cpu(),s=1)
#plt.scatter(xt_resid[0:int(np.sqrt(N_simple)),1].detach().cpu(),xt_resid[0:int(np.sqrt(N_simple)),0].detach().cpu(),s=10)

IC_data = create_IC_data(Xi, Xf, Ti, Tf, IC_pts, IC_simple)
IC_xt     = IC_data[0].to(device)
IC_u      = IC_data[1].to(device)
#plt.scatter(IC_xt[:,1].detach().cpu(),IC_xt[:,0].detach().cpu(),s=1)

BC_data = create_BC_data(Xi, Xf, Ti, Tf, BC_pts)
BC_bottom     = BC_data[0].to(device)
BC_top      = BC_data[1].to(device)

total_train_time_1 = time.perf_counter()
print("******************************************************************")
########################### Full PINN Training ############################    
pinn_train_time_1 = time.perf_counter()
PINN = NN(layers_pinn, nu_pinn_train).to(device)

pinn_losses = pinn_train(PINN, nu_pinn_train, xt_resid, IC_xt, IC_u, BC_bottom, BC_top, f_hat, epochs_pinn, lr_pinn, tol, xt_test)

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
        
gpt_activation = P(layers_pinn, w1, w2, w3,w4, b1, b2, b3,b4).to(device)

lr_tgpt          = 0.05
epochs_tgpt      = 100000
tol_tgpt         = 1e-5
layers_tgpt      = [2,1,1]

#################### Training TGPT-PINN ######################
nu = -10.0
xt_train_nu = create_residual_data(nu, Xi, Xf, Ti, Tf, N_train, N_test, N_simple)[0].to(device)
#plt.scatter(xt_train_nu[0:int(np.sqrt(N_simple)),1].detach().cpu(),xt_train_nu[0:int(np.sqrt(N_simple)),0].detach().cpu(),s=10)
TGPT_PINN = GPT(layers_tgpt, nu, gpt_activation, f_hat, IC_u, xt_train_nu, IC_xt, BC_bottom, BC_top).to(device)
    
tgpt_losses = gpt_train(TGPT_PINN, nu, f_hat, IC_u, xt_train_nu, xt_test, IC_xt, BC_bottom, BC_top, epochs_tgpt, lr_tgpt, tol_tgpt)
    
Trans_plot(xt_test, TGPT_PINN.forward(xt_test), title=fr"$\nu={round(nu,1)}$")
Trans_plot(xt_test, exact_u(xt_test,nu), title=fr"$\nu={round(nu,1)}$")
Trans_plot(xt_test,abs(TGPT_PINN.forward(xt_test)-exact_u(xt_test,nu)),title=fr"$\nu={round(nu,1)}$")
loss_plot(tgpt_losses[2], tgpt_losses[1], title=fr"$\nu={round(nu,1)}$")