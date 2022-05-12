'''
Implementation of paper 'Joint-Label Learning by Dual Augmentation for Time Series Classification'
Author: Zhengke Sun
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class tws_item(nn.Module):

    def __init__(self, N):
        super(tws_item, self).__init__()

        self.avg = nn.AvgPool1d(kernel_size=1,stride=2)
        self.N =N
        assert self.N%2 == 0 , 'N must be even !!'

    def forward(self,x):

        _, _, l = x.shape
        x_aug = torch.tensor([])
        x_split = np.split(x, self.N, axis=2)

        for i in range(0, self.N, 2):
            x_aug = torch.cat([x_aug,
                               self.avg(x_split[i]),
                               F.interpolate(x_split[i + 1], int(l/self.N+l/self.N//2), mode='linear',
                                             align_corners=False)],
                               dim=2)

        return x_aug

class tws(nn.Module):
    def __init__(self,item,N_list):
        super(tws, self).__init__()

        self.tws_item = item
        self.N_list = N_list

    def forward(self,x):
        stack_x_aug = x

        for i in range(len(self.N_list)):
            tws_out = self.tws_item(N=self.N_list[i])
            stack_x_aug = torch.cat([stack_x_aug,tws_out(x)], dim=0)

        return stack_x_aug

if __name__ == '__main__':

    input=Variable(torch.randn(128,3,40))
    tws_i = tws_item(N=4)
    x_aug = tws_i(input)
    print(x_aug.shape)

    tws = tws(tws_item, N_list=[4,8,10,20])
    stack_x_aug = tws(input)
    print(stack_x_aug.shape)

