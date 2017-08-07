from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet | folder ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--network', default='', help="path to wide residual network (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--valid', default=False, help='"valid" being true, we do validation only')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

"""
if opt.dataset in ['mnist', 'MNIST']:
    # folder dataset
    dataset = dset.MNIST(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                            	   transforms.Normalize((0.5,), (0.5,)),
                               ]))



if opt.dataset in ['imagenet', 'folder']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                            	   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
"""
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
trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
validloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
ngpu = int(opt.ngpu)
#nc = 3
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
        self.group0.conv11 = nn.Conv2d(hexa, hexa*k, 1, 1, 0, bias=False)
        self.group0.block0 = nn.Sequential(
            # input is hexa x 32 x 32
            nn.BatchNorm2d(hexa),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
        self.group0.block1 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
        self.group0.block2 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
            # state size. (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k, 3, 1, 1, bias=False),
        )
        self.group1.block0 = nn.Sequential(
            # input is (hexa*k) x 32 x 32
            nn.BatchNorm2d(hexa*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k, hexa*k*2, 3, 2, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )
            # input ie (hexa*k) X 32 X 32
        self.group1.conv11 = nn.Conv2d(hexa*k, hexa*k*2, 1, 2, 0, bias=False)

        self.group1.block1 = nn.Sequential(
            # input is (hexa**2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )
        self.group1.block2 = nn.Sequential(
            # input is (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
            # state size. (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*2, 3, 1, 1, bias=False),
        )

        self.group2.block0 = nn.Sequential(
            # input is (hexa*k*2) x 16 x 16
            nn.BatchNorm2d(hexa*k*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*2, hexa*k*4, 3, 2, 1, bias=False),
            # state size. (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
        )
            # input ie (hexa*k*2) X 16 X 16
        self.group2.conv11 = nn.Conv2d(hexa*k*2, hexa*k*4, 1, 2, 0, bias=False)

        self.group2.block1 = nn.Sequential(
            # input is (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
            # state size. (hexa*k*4) x 8 x 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hexa*k*4, hexa*k*4, 3, 1, 1, bias=False),
        )
        self.group2.block2 = nn.Sequential(
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
        self.top = nn.Liear(hexa*k*4, numClass,bias=True)

    def forward(self, input):
        #conv0
        output = self.conv0(input)
        #group0
        residual = self.group0.block0(output)
        straight = self.group0.conv11(output)
        output = self.group0.block1(residual+straight)
        output = self.group0.block2(output)+output
        #group1
        residual = self.group1.block0(output)
        straight = self.group1.conv11(output)
        output = self.group1.block1(residual+straight)
        output = self.group1.block2(output)+output
        #group2
        residual = self.group2.block0(output)
        straight = self.group2.conv11(output)
        output = self.group2.block1(residual+straight)
        output = self.group2.block2(output)+output
        #top
        output = self.avg(output)
        output = self.top(output.view(output.size(0), -1))
        return output


WRN = _WRN(numClass, hexa, k)
WRN.apply(weights_init)
if opt.network != '':
    network.load_state_dict(torch.load(opt.network))
print(network)

criterion = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize, numClass)


if opt.cuda:
    WRN.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()

# setup optimizer
optimizerD = optim.Adam(WRN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
def validation():
    correct = 0
    for i, data in enumerate(validloader):
        WRN.eval()
        input_cpu, label_cpu = data
        batch_size = input_cpu.size(0)
        if opt.cuda:
            input_gpu = input_cpu.cuda()
            label_gpu = label_cpu.cuda()
            input.resize_as_(input_gpu).copy_(input_gpu)
            label.resize_as_(label_gpu).copy_(label_gpu)
        else:
            input.resize_as_(input_cpu).copy_(input_cpu)
            label.resize_as_(label_cpu).copy_(label_cpu)
        inputv = Variable(input)
        labelv = Variable(label)
         if opt.ngpu > 1:
           output = nn.parallel.data_parallel(WRN, inputv, range(opt.ngpu))
        else :
           output = WRN(inputv)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
     return 100*correct / (len(validloader)*batch_size)
                           
                           
#validation only!                           
if opt.valid == True:
    accuracy = validation()
    print('validation Accuracy: %.4f'
              % (accuracy))
    return
                           
                           
for epoch in range(opt.niter):
    
    for i, data in enumerate(trainloader):
        ###########################
        ###########################
        WRN.train()
        WRN.zero_grad()
        input_cpu, label_cpu = data
        batch_size = input_cpu.size(0)
        if opt.cuda:
            input_gpu = input_cpu.cuda()
            label_gpu = label_cpu.cuda()
            input.resize_as_(input_gpu).copy_(input_gpu)
            label.resize_as_(label_gpu).copy_(label_gpu)
        else:
            input.resize_as_(input_cpu).copy_(input_cpu)
            label.resize_as_(label_cpu).copy_(label_cpu)
        inputv = Variable(input)
        labelv = Variable(label)
        if opt.ngpu > 1:
           output = nn.parallel.data_parallel(WRN, inputv, range(opt.ngpu))
        else :
           output = WRN(inputv)
        
        err = criterion(output, labelv)
        err.backward()
        optimizerD.step()

       
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, opt.niter, i, len(trainloader),
                 err.data[0]))
        if i % 100 == 0:
            accuracy = validation()
            print('validation Accuracy: %.4f'
              % (accuracy))
                 
    # do checkpointing
    torch.save(WRN.state_dict(), '%s/WRN_epoch_%d.pth' % (opt.outf, epoch))
