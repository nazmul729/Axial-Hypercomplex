##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy                   as np
from   numpy.random            import RandomState
import torch
from torch import Tensor
from   torch.autograd           import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from   torch.nn.parameter       import Parameter
from   torch.nn                 import Module, init
#from   .quaternion_ops          import *
import math
import sys


class QLinear(nn.Module):

  def __init__(self, n, in_features, out_features):
    super(QLinear, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features

    self.bias = Parameter(torch.Tensor(out_features))

    self.a = torch.zeros((n, n, n))
    self.a = Parameter(torch.nn.init.xavier_uniform_(self.a))

    self.s = torch.zeros((n, self.out_features//n, self.in_features//n)) 
    self.s = Parameter(torch.nn.init.xavier_uniform_(self.s))

    self.weight = torch.zeros((self.out_features, self.in_features))

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)

  def kronecker_product1(self, a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    #print("siz1: ",siz1)
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    #print("Res: ",res.shape)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def forward(self, input: Tensor) -> Tensor:
      #print("In and Out Features: ",self.in_features, self.out_features)
      #print("Weight before: ", self.weight.shape)
      self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
      #print("Weight after: ",self.weight.shape, input.shape)
      input = input.type(dtype=self.weight.type())
      return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
      return 'in_features={}, out_features={}, bias={}'.format(
          self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
      init.kaiming_uniform_(self.a, a=math.sqrt(5))
      init.kaiming_uniform_(self.s, a=math.sqrt(5))
      fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
      bound = 1 / math.sqrt(fan_in)
      init.uniform_(self.bias, -bound, bound)