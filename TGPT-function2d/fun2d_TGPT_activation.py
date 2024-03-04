import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class P(nn.Module): 
    def __init__(self, P_func_nu):
        """TGPT-PINN Activation Function"""
        super().__init__()
        self.layers = [1, 1, 1]

        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1], bias=False) for i in range(len(self.layers)-1)])
        
        #self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        #self.linears[1].weight.data = torch.ones(self.layers[2], self.layers[1])
        
        self.activation = P_func_nu

    def forward(self, x): 
        a = self.activation(x.to(device)) 
        #a = x.to(device)
        #for i in range(0, len(self.layers)-2):
            #z = self.linears[i](a)
            #a = self.activation(z)        
        #a = self.linears[-1](a)
        return a
