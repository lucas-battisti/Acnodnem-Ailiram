from torch import nn
import torch

class MultiInputSequential(nn.Module):
    def __init__(self, *modules):
        super(MultiInputSequential, self).__init__()
        self.modules = modules
        
    def forward(self, *inputs):
        for module in self.modules:
            if type(inputs) == tuple:
                if len(inputs) == 1:
                    inputs = module(inputs[0])
                else:
                    inputs = module(inputs)
            else:
                inputs = module(inputs)
        return inputs

class MultiModule(nn.Module):
    def __init__(self, *modules):
        super(MultiModule, self).__init__()
        self.modules = modules
        
    def forward(self, inputs: tuple):
        if len(self.modules) != len(inputs):
            raise TypeError("Number of inputs are incompatible with numer of modules")
        
        return_tuple = tuple()
        
        for i in range(len(inputs)):
             return_tuple += (self.modules[i](inputs[i]),)
             
        return return_tuple
        
class Slicer(nn.Module):
    def __init__(self, last_k: int = 1, dim: int=0):
        super(Slicer, self).__init__()
        self.last_k = last_k
        self.dim = dim
        
    def forward(self, input):
        
        length = input.size(self.dim)
        
        x1 = torch.narrow(input, self.dim, 0, length-self.last_k)
        x2 = torch.narrow(input, self.dim, length-1-self.last_k, self.last_k)
        
        return x1, x2
    
class Joiner_Flatten(nn.Module):
    def __init__(self):
        super(Joiner_Flatten, self).__init__()
        
    def forward(self, x: tuple):
        x1 = x[0]
        x2 = x[1]
        
        x1 = nn.Flatten(1)(x1)
        
        dims = tuple(range(1, x2.dim()-1))
        x2 = torch.mean(x2, dims, True)
        x2 = torch.reshape(x2, (x2.size()[0], x2.size()[-1]))
        
        return torch.cat((x1, x2), 1)
        


