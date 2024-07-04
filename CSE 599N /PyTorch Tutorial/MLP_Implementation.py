
import torch.nn as nn # Base Class that we inherit from 
import torch.optim as optim
import torch

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, act=nn.ReLU): # Class that makes MLPs for us, nn.ReLU = our activation function
        super().__init__() # Calls the constructor for the base class
        self.weight = nn.Parameter(torch.randn(in_size, out_size))

        net = [] # List of layers 
        prev_dim = in_size
        for hidden in hidden_size: 
            layer = nn.Linear(prev_dim, hidden) # Creates a linear layer that does a weighted sum + bias 
            net.append(layer)
            net.append(act())  # Creating an instance of the activation function (is itself is an nn.Module)
            prev_dim = hidden # Output becomes input of the next layer

        layer = nn.Linear(prev_dim, out_size) # Takes output of last hidden layer and maps to hidden size of our entire network
        net.append(layer)
        self.net = nn.Sequential(net) # Executes each layer in order of our list, passing the output of the previous layer to the next 
    
    def forward(self, x):
        return self.net(x)
    
    def predict(self, x): # Returns integer corresponding to the predicted class 
        output = self.forward(x)
        return torch.argmax(output, dim = 1)