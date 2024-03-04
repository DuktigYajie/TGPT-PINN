import torch
torch.set_default_dtype(torch.float)

def gpt_train(TGPT_PINN, layers_gpt, nu, P, train_xy, u_exact, epochs_gpt, lr_gpt, tol_gpt):

    optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_gpt)

    loss_values = TGPT_PINN.loss()
    losses=[loss_values.item()]
    ep=[0]

    for i in range(1, epochs_gpt+1):
        if (loss_values < tol_gpt): 
            losses.append(loss_values.item())
            ep.append(i)
            print(f'{nu} stopped at epoch: {i} | gpt_loss: {loss_values.item()} (TGPT_PINN Stopping Criteria Met)\n')
            break
                
        else:
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

        loss_values = TGPT_PINN.loss()

        if (i % 50 == 0) or (i == epochs_gpt):
            losses.append(loss_values.item())
            ep.append(i)
        if (i == epochs_gpt):
            print(f"{nu} stopped at epoch: {i} | gpt_loss: {loss_values.item()} (TGPT-PINN Training Completed)\n")
                    

    return loss_values, losses, ep