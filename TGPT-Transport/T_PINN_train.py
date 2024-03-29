import torch
torch.set_default_dtype(torch.float)    

def pinn_train(PINN, nu, xt_resid, IC_xt, IC_u, BC1, BC2,
               f_hat, epochs_pinn, lr_pinn, tol, xt_test):
    
    losses = [PINN.loss(xt_resid, IC_xt, IC_u, BC1, BC2, f_hat).item()] 
    ep = [0]
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    
    print(f"Epoch: 0 | Loss: {losses[0]}")
    for i in range(1, epochs_pinn+1):
        loss_values = PINN.loss(xt_resid, IC_xt, IC_u, BC1, BC2, f_hat)
        
        if (loss_values.item() < tol):
            losses.append(loss_values.item())
            ep.append(i)
            print(f'Epoch: {i} | Loss: {loss_values.item()} (Stopping Criteria Met)')
            break
        
        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        
        if (i % 1000 == 0) or (i == epochs_pinn):
            losses.append(loss_values.item())
            ep.append(i)
            if (i % 5000 == 0) or (i == epochs_pinn):
                print(f'Epoch: {i} | loss: {loss_values.item()}')
                if (i == epochs_pinn):
                    print("PINN Training Completed\n")
                                            
    return losses, ep, loss_values
