import torch
torch.set_default_dtype(torch.float)

def gpt_train(TGPT_PINN, nu, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_gpt, lr_gpt, tol_gpt):

    optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_gpt)
    
    losses=[TGPT_PINN.loss().item()]
    ep=[0]
    
    change_tol = tol_gpt*100

    #print(f"Epoch: 0 | tgpt-Loss: {losses[0]}")
    #print(f"layer1:{TGPT_PINN.linears[0].weight.data} and layer2:{TGPT_PINN.linears[1].weight.data}")
    for i in range(1, epochs_gpt+1):
        loss_values = TGPT_PINN.loss()
        
        if (loss_values.item() < tol_gpt):
            losses.append(loss_values.item())
            ep.append(i)
            print(f'{round(nu,3)} stopped at epoch: {i} | Loss: {loss_values.item()} (TGPT_PINN Stopping Criteria Met)')
            break
        
        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        
        TGPT_PINN.linears[0].weight.data[0,0] = TGPT_PINN.linears[0].weight.data[0,0]/TGPT_PINN.linears[0].weight.data[0,0]
        TGPT_PINN.linears[0].weight.data[0,1] = TGPT_PINN.linears[0].weight.data[0,1]/TGPT_PINN.linears[0].weight.data[0,0]
        if (nu*TGPT_PINN.linears[0].weight.data[0,1] > 0):
            TGPT_PINN.linears[0].weight.data[0,1] = - TGPT_PINN.linears[0].weight.data[0,1]
            
        if (i % 500 == 0) or (i == epochs_gpt):
            losses.append(loss_values.item())
            ep.append(i)
            if (i % 5000 == 0) or (i == epochs_gpt):
                print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values.item()}')
                #print(f"layer1:{TGPT_PINN.linears[0].weight.data}and layer2:{TGPT_PINN.linears[1].weight.data}")
                if (i == epochs_gpt):
                    print("TGPT-PINN Training Completed")
                    
        if (loss_values.item() < change_tol):
            lr_gpt = 0.1*lr_gpt
            change_tol = 0.1*change_tol

    #print(f"layer1:{TGPT_PINN.linears[0].weight.data} and layer2:{TGPT_PINN.linears[1].weight.data}\n")
    return loss_values, losses, ep
