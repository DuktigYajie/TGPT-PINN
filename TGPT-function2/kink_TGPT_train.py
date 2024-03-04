import torch
torch.set_default_dtype(torch.float)

def gpt_train(TGPT_PINN, layers_gpt, nu,epochs_gpt, lr_gpt, x_tol_gpt,u_tol_gpt):

    optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_gpt)

    x_loss_values = TGPT_PINN.loss_x()
    u_loss_values = TGPT_PINN.loss_u()
    x_losses=[x_loss_values.item()]
    u_losses=[u_loss_values.item()]
    x_ep=[0]
    u_ep=[0]

    for i in range(1, epochs_gpt+1):
        if (x_loss_values < x_tol_gpt): 
            x_losses.append(x_loss_values.item())
            x_ep.append(i)
            break
                
        else:
            optimizer.zero_grad()
            x_loss_values.backward()
            optimizer.step()

        x_loss_values = TGPT_PINN.loss_x()

        if (i % 50 == 0) or (i == epochs_gpt):
            x_losses.append(x_loss_values.item())
            x_ep.append(i)

    for i in range(1, epochs_gpt+1):        
        if (u_loss_values < u_tol_gpt): 
            u_losses.append(u_loss_values.item())
            u_ep.append(i)
            break
                
        else:
            optimizer.zero_grad()
            u_loss_values.backward()
            for j in range(0,layers_gpt[1]):
                TGPT_PINN.linears[j].weight.grad = None
                TGPT_PINN.linears[j].bias.grad = None
            optimizer.step()

        u_loss_values = TGPT_PINN.loss_u() 

        if (i % 50 == 0) or (i == epochs_gpt):
            u_losses.append(u_loss_values.item())
            u_ep.append(i)


    #for j in range(0,layers_gpt[1]):
    #    print(f"layer{j}_weight: {TGPT_PINN.linears[j].weight.data.item()} and bias: {TGPT_PINN.linears[j].bias.data.item()} ")
    #print(f"layer_ouput:{TGPT_PINN.linears[-1].weight.data}")   
    x_loss_values = TGPT_PINN.loss_x()
    u_loss_values = TGPT_PINN.loss_u()
    print(f"{nu} stopped at epoch: {i} | x_loss: {x_loss_values.item()} and u_loss:{u_loss_values.item()}\n")
    return x_loss_values, u_loss_values, x_losses, u_losses, x_ep, u_ep
