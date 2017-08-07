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
parser.add_argument('--network', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

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
"""

if opt.dataset in ['imagenet', 'folder']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                            	   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nc = 3
hexa = 16
k = 10 

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class _WRN(nn.Module):
    def __init__(self, ngpu, numClass, hexa, k):
        super(_WRN, self).__init__()
            # input is (nc) x 32 x 32
        self.conv0 = nn.Conv2d(nc, hexa, 3, 1, 1, bias=False)  
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

        self.top = nnSequential(
            # input is (hexa*k*4) X 8 X 8
            nn.BatchNorm2d(hexa*k*4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
            nn.Linear(hexa*k*4, 10),
        )

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
        output = self.top(output)
        return output.view(-1, 1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
