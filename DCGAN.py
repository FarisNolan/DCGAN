# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:03:23 2018

@author: Faris
"""

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#------------------Custom weights--------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
 #--------------Generator-------------       
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            #State size (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            #State size (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            #State Size (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            #State size ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
            #State size nc x 64 x 64
            )
        
    def forward(self, input):
        return self.main(input)

#---------------Discriminator------------
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
                )
        
    def forward(self, input):
        return self.main(input)
    
#----------------Set up values------------------
        
# Root directory for dataset
dataroot = r'''C:\Users\Faris\Documents\GitHub\datasets\img_align_celeba'''
    
# Number of workers for dataloader
workers = 2
    
# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

def main():
    #Set random seed
    manualSeed = 999
    #manualSeed = random.randint(1, 1000) #if you want new results
    print("Random seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    
    
    #--------------------Set up data-------------------
    #Can use image folder, due to how it is set up
    dataset = dset.ImageFolder(root = dataroot, transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
    
    #--------------create the dataloader-------------
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)
    
    #handle device to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print (device)
    print(torch.backends.cudnn.enabled)
    
    #-------------plot some training examples-----------
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    #----------create the generator--------------
    netG = Generator(ngpu).to(device)
    
    #Handle multi GPU
    if (device.type == "cuda") and (ngpu < 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    
    #Apply weights_init to randomly intialize all weights in netG
    #mean = 0, std = 0.2
    netG.apply(weights_init)

    #print model
    print(netG)
        
    #-------------create the discriminator-------------
    netD = Discriminator(ngpu).to(device)
    
    #handle multi GPU
    if (device.type == "cuda") and (ngpu < 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    #Apply weights_init to randomly intialize all weights in netD
    #mean = 0, std = 0.2
    netD.apply(weights_init)
    
    print(netD)
    
    #-----------set up loss function--------------
    criterion = nn.BCELoss()
    
    #batch of random noise vectors that will be used for progression visualization of G
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    #Establish convention for real and fake labels
    real_label = 1
    fake_label = 0
    
    #setup Adam optimizer for G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    #--------------Training Loop-------------
    #Progress trackers
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting training loop...")
    #For each epoch
    for epoch in range(num_epochs):
        #For each batch
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            
            ## Train with all-real batch
            netD.zero_grad()
            #Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            #Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            #Loss on all real batch
            errD_real = criterion(output, label)
            #Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            #Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device = device)
            #Generate fake image batch
            fake = netG(noise)
            label.fill_(fake_label)
            #Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            #D's Loss on all-fake batch
            errD_fake = criterion(output, label)
            #Gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            #Sum gradients
            errD = errD_real + errD_fake
            #Update D
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            #G's loss based on output
            errG = criterion(output, label)
            #Gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            #update G
            optimizerG.step()
            
            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            #Save losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise,).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
                
    #-----------------Plot Progression---------------
    #Loss over iterations
    plt.figure(figsize=(10,5))
    plt.title("G and D Loss Over Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    #Generator over time
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    
    #Real vs fake images
    #grab a batch
    real_batch = next(iter(dataloader))
    #plot real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("real images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    #Fake images from last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("fake images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
    
if __name__ == "__main__":
    main()








