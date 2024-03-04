import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
plt.style.use(['science', 'notebook'])

def Trans_plot(xt, u, scale=150, cmap="rainbow", title=None, 
                 dpi=150, figsize=(10,8)):
    
    shape = [int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0]))]
    
    x = xt[:,0].reshape(shape=shape).transpose(1,0).cpu().detach() 
    t = xt[:,1].reshape(shape=shape).transpose(1,0).cpu().detach() 
    u =       u.reshape(shape=shape).transpose(1,0).cpu().detach()
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    cp = ax.contourf(t, x, u, scale, cmap=cmap)
    cbar = fig.colorbar(cp)
        
    ax.set_xlabel("$t$", fontsize=25)
    ax.set_ylabel("$x$", fontsize=25)
    
    ax.set_xticks([ 0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([-1.0, -0.5,  0.0,  0.5, 1.0])
    
    ax.tick_params(axis='both', which='major', labelsize=22.5)
    ax.tick_params(axis='both', which='minor', labelsize=22.5)
    cbar.ax.tick_params(labelsize=17.5)
    
    if title is not None:
        ax.set_title(title, fontsize=30)
        
    plt.show()

def loss_plot(epochs, losses, title=None, dpi=150, figsize=(10,8)):
    """Training losses"""
    plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(epochs, losses, c="k", linewidth=3)
    
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss", fontsize=20)
     
    plt.grid(True)
    plt.xlim(0,max(epochs))
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=22.5)
    if title is not None:
        plt.title(title,fontsize=30)
    
    plt.show() 

