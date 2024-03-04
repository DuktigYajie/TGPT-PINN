import torch
torch.set_default_dtype(torch.float)

def gpt_train(TGPT_PINN, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_gpt, lr_gpt, tol_gpt,testing=False):
    
    optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_gpt)

    loss_values = TGPT_PINN.loss()
    losses=[loss_values.item()]
    ep=[0]

    #print(f"Epoch: 0 | tgpt_loss: {losses[0]}")
    #print(f"layer1:{TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[0].bias.data} and layer3:{TGPT_PINN.linears[-1].weight.data}")
    #print(f"layer1:{TGPT_PINN.linears[0].weight.data}and layer3:{TGPT_PINN.linears[-1].weight.data}")
    #print(f'{round(rho,3)} at epoch: 1 | gpt_loss: {loss_values.item()}')
    if (testing == False): 
        loss_values = TGPT_PINN.loss()
        for i in range(1, epochs_gpt+1):
            if (loss_values < tol_gpt): 
                losses.append(loss_values.item())
                ep.append(i)
                #print(f"layer1:{TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[0].bias.data} and layer3:{TGPT_PINN.linears[-1].weight.data}")
                print(f'{round(rho,3)} stopped at epoch: {i} | gpt_loss: {loss_values.item()} (TGPT_PINN Stopping Criteria Met)')
                break
                
            else:
                optimizer.zero_grad()
                loss_values.backward()
                optimizer.step()
        
            loss_values = TGPT_PINN.loss()

            if (i % 500 == 0) or (i == epochs_gpt):
                losses.append(loss_values.item())
                ep.append(i)
                if (i % 5000 == 0) or (i == epochs_gpt):
                    print(f'{round(rho,3)} stopped at epoch: {i} | tgpt_loss: {loss_values.item()}')
                    #print(f"layer1:{TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[0].bias.data} and layer3:{TGPT_PINN.linears[-1].weight.data}")
                    if (i == epochs_gpt):
                        print("TGPT-PINN Training Completed")                
        return loss_values, losses, ep
    
    elif (testing):
        loss_values = TGPT_PINN.loss()
        losses=[loss_values.item()]
        ep=[0]
        for i in range(1, epochs_gpt+1):
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()
            loss_values = TGPT_PINN.loss()
            if (i % 200 == 0) or (i == epochs_gpt):
                losses.append(loss_values.item())
                ep.append(i)
        loss_values = TGPT_PINN.loss()
        return loss_values,losses,ep