from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
import matplotlib.pyplot as P
import torch.nn.functional as F
from IPython.core.debugger import Pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--network', default='', help="path to wide residual network (to continue training)")
opt = parser.parse_args()
print(opt)



#define datasets
trainset = dset.CIFAR10(root=opt.dataroot, train = True, download=True, 
                           transform=transforms.Compose([
                           transforms.Scale(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
validset = dset.CIFAR10(root=opt.dataroot, train = False, download=True, 
                           transform=transforms.Compose([
                           transforms.Scale(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )    
assert trainset
assert validset

#define loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
validloader = torch.utils.data.DataLoader(validset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
numClass = 10
hexa = 16
k = 10 

# custom weights initialization called on WRN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class _WRN(nn.Module):
    def __init__(self, numClass, hexa, k):
        super(_WRN, self).__init__()
            # input is (nc) x 32 x 32
        self.conv0 = nn.Conv2d(3, hexa, 3, 1, 1, bias=False)  
            # input is (hexa) X 32 x 32
        self.group0_conv11 = nn.Conv2d(hexa, hexa*k, 1, 1, 0, bias=False)
        self.group0_block0 = nn.Sequential(
            # input is hexa x 32 x 32
            nn.BatchNorm2d(hexa),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
        self.group0_block1 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
        self.group0_block2 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
            # input is (hexa*k) X 32 X 32
        self.group1_conv11 = nn.Conv2d(hexa*k, hexa*k*2, 1, 2, 0, bias=False)
        self.group1_block0 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k*2, 3, 2, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )
        self.group1_block1 = nn.Sequential(
            # input is (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )
        self.group1_block2 = nn.Sequential(
            # input is (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )
            # input is (hexa*k*2) X 16 X 16
        self.group2_conv11 = nn.Conv2d(hexa*k*2, hexa*k*4, 1, 2, 0, bias=False)
        self.group2_block0 = nn.Sequential(
            # input is (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*4, 3, 2, 1, bias=False),
            # state size. (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
        )
       
        self.group2_block1 = nn.Sequential(
            # input is (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
            # state size. (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
        )
        self.group2_block2 = nn.Sequential(
            # input is (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
            # state size. (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
        )

        self.avg = nn.Sequential(
            # input is (hexa*k*4) X 8 X 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8),           
        )
        self.top = nn.Linear(hexa*k*4, numClass,bias=True)

    def forward(self, input):
        #conv0
        output = self.conv0(input)
        #group0
        residual = self.group0_block0(output)
        straight = self.group0_conv11(output)
        output = self.group0_block1(residual+straight)
        output = self.group0_block2(output)+output
        #group1
        residual = self.group1_block0(output)
        straight = self.group1_conv11(output)
        output = self.group1_block1(residual+straight)
        output = self.group1_block2(output)+output
        #group2
        residual = self.group2_block0(output)
        straight = self.group2_conv11(output)
        output = self.group2_block1(residual+straight)
        output = self.group2_block2(output)+output
        #top
        output = self.avg(output)
        output = self.top(output.view(output.size(0), -1))
        return output


WRN = _WRN(numClass, hexa, k)
WRN.apply(weights_init)
if opt.network != '':
    WRN.load_state_dict(torch.load(opt.network))
print(WRN)

criterion = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.LongTensor(opt.batchSize, numClass)



# setup optimizer

Pdb().set_trace()
for i, data in enumerate(validloader):
    WRN.eval()
    input_cpu, label_cpu = data
    batch_size = input_cpu.size(0)
    input.resize_as_(input_cpu).copy_(input_cpu)
    label.resize_as_(label_cpu).copy_(label_cpu)
    inputv = Variable(input)
    labelv = Variable(label)
    out1 = F.conv2d(inputv, WRN.conv0.weight,WRN.conv0.bias,1,1)
    out2 = F.conv2d(out1, WRN.group0_block0[2].weight,WRN.group0_block0[2].bias ,1,1)
    output = F.conv2d(out2,
                      WRN.group0_block0[5].weight,WRN.group0_block0[5].bias,1,1)

    Pdb().set_trace()
                           
