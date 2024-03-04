import torch
torch.set_default_dtype(torch.float)

def gpt_train(TGPT_PINN, nu, rho, f_hat, IC_u, xt_resid, xt_test, IC_xt, BC1, BC2, epochs_gpt, lr_gpt,
              largest_loss=None, largest_case=None,testing=False):
    
    optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_gpt)

    if (testing == False): 
        loss_values = TGPT_PINN.loss()
        for i in range(1, epochs_gpt+1):
            if (loss_values < largest_loss): 
                break
                
            else:
                optimizer.zero_grad()
                loss_values.backward()
                optimizer.step()
                if (i == epochs_gpt):
                    largest_case = [nu,rho]
                    largest_loss = TGPT_PINN.loss() 
            loss_values = TGPT_PINN.loss()

        return largest_loss,largest_case

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

        return loss_values,losses,ep