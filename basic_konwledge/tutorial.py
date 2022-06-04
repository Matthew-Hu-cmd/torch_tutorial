import torch
from torch import nn, optim
import numpy

a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 1*28*28))  #

