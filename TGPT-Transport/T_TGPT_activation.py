import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class P(nn.Module):
    def __init__(self, layers, w1, w2, w3,w4, b1, b2, b3,b4):
        super().__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        
        self.linears[0].weight.data = torch.Tensor(w1).clone()
        self.linears[1].weight.data = torch.Tensor(w2).clone()
        self.linears[2].weight.data = torch.Tensor(w3).clone()
        self.linears[3].weight.data = torch.Tensor(w4).clone().view(1,self.layers[3])

        self.linears[0].weight.requires_grad = False
        self.linears[1].weight.requires_grad = False
        self.linears[2].weight.requires_grad = False
        self.linears[3].weight.requires_grad = False

        self.linears[0].bias.data = torch.Tensor(b1).clone()
        self.linears[1].bias.data = torch.Tensor(b2).clone()
        self.linears[2].bias.data = torch.Tensor(b3).clone()
        self.linears[3].bias.data = torch.Tensor(b4).clone().view(-1)

        self.linears[0].bias.requires_grad = False
        self.linears[1].bias.requires_grad = False
        self.linears[2].bias.requires_grad = False
        self.linears[3].bias.requires_grad = False
        
        self.activation = nn.Tanh()
        
    def forward(self, x):      
        """TGPT-PINN Activation Function"""
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)        
        a = self.linears[-1](a)
        return a
