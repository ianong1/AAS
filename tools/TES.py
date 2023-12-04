import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Module
import torch.nn as nn

class TES(nn.Module):
    def __init__(self,tau=0.1):
        super(TES,self).__init__()
        self.tau = tau

    def forward(self,p1,m1):
        seed = np.random.randint(10000000)
        torch.manual_seed(seed)
        modi_pattern = torch.tensor([[1],[0],[-1]]).float().type_as(p1)
        logits = torch.log(torch.stack((p1,1-p1-m1,m1),dim=4))
        y = F.gumbel_softmax(logits,tau=self.tau,hard=True,dim=4)
        b = torch.einsum('abcde,ef->abcdf',y,modi_pattern)

        return b.squeeze(4)

