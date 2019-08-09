import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import os


data_path='C:\\Users\\Ayush\\Data\\'
image_size=64
gfd=64
dfd=64
nc=3
batch_size=128
workers=1
epochs=5
lr=0.0002


dataset=dset.ImageFolder(root=data_path,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]))

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

device=torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

real_batch=next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64],padding=2,normalize=True),(1,2,0)))


plt.imshow(np.transpose(real_batch[0][1],(1,2,0)))

def init_weight(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        nn.init.normal(m.weight.data,0.0,0.2)
    elif classname.find("BatchNorm")!=-1:
        nn.init.normal(m.weight.data,0.0,0.2)
        nn.init.constant(m.bias.data,0)
        
nz=100
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(nz,gfd*8,4,1,0,bias=False),
            nn.BatchNorm2d(8*gfd),
            nn.ReLU(True),
            nn.ConvTranspose2d(gfd*8,gfd*4,4,1,0,bias=False),
            nn.BatchNorm2d(4*gfd),
            nn.ReLU(True),
            nn.ConvTranspose2d(gfd*4,gfd*2,4,1,0,bias=False),
            nn.BatchNorm2d(2*gfd),
            nn.ReLU(True),
            nn.ConvTranspose2d(gfd*2,gfd,4,1,0,bias=False),
            nn.BatchNorm2d(gfd),
            nn.ReLU(True),
            nn.ConvTranspose2d(gfd,nc,4,2,1,bias=False),
            nn.Tanh()
            
        )
    def forward(self,input):
        return self.main(input)

netG=Generator().to(device)
netG.apply(init_weight)
print(netG)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
        nn.Conv2d(nc,dfd,4,2,1,bias=False),
        nn.LeakyReLU(0.2,inplace=True),
            
        nn.Conv2d(dfd,2*dfd,4,2,1,bias=False),
        nn.BatchNorm2d(2*dfd),
        nn.LeakyReLU(0.2,inplace=True),

        nn.Conv2d(2*dfd,4*dfd,4,2,1,bias=False),
        nn.BatchNorm2d(4*dfd),
        nn.LeakyReLU(0.2,inplace=True),
            
        nn.Conv2d(4*dfd,8*dfd,4,2,1,bias=False),
        nn.BatchNorm2d(8*dfd),
        nn.LeakyReLU(0.2,inplace=True),
            
        nn.Conv2d(8*dfd,1,4,1,0,bias=False),
        nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)


netD=Discriminator().to(device)
netD.apply(init_weight)
print(netD)

criterion=nn.BCELoss()
optimizerG=Adam(netG.parameters(),lr=lr,betas=(0.5,0.999))
optimizerD=Adam(netD.parameters(),lr=lr,betas=(0.5,0.999))
real_label=1
fake_label=0

#Noise Vector for Image Generation
noise=torch.randn(64,nz,1,1,device=device)

G_losses,D_losses=[],[]
img_list=[]
iters=0

print("Starting Training.....")
for e in range(epochs):
    for i,images in enumerate(dataloader,0):
        images=images[0]
        netD.zero_grad()
        optimizerD.zero_grad()
        images=images.to(device)
        label=torch.full((images.shape[0],),real_label,device=device)
        output=netD(images).view(-1)
        errorD_R=criterion(output,label)
        errorD_R.backward()
        D_x=output.mean().item()
        
        #Now D(G(z))
        noise=torch.randn(images.shape[0],100,1,1,device=device)
        fake=netG(noise)
        label.fill_(fake_label)
        output=netD(fake.detach()).view(-1)
        errorD_F=criterion(output,label)
        errorD_F.backward()
        D_G_z1=output.mean().item()
        errorD=errorD_R+errorD_F
        optimizerD.step()
        
        #Now D(G(z')) For updating Generator
        netG.zero_grad()
        optimizerG.zero_grad()
        label.fill_(real_label)
        output=netD(fake.detach()).view(-1)
        errorG=criterion(output,label)
        errorG.backward()
        D_G_z2=output.mean().item()
        optimizerG.step()
        
        if i%50==0:
            print("Epoch:{}/{} {}/{} LossD {:.3f} LossG {:.3f} d_x {:.2f} d_g_z1 {:.2f} d_g_z2 {:.2f}".format(e+1,epochs,i,len(dataloader),errorD.item(),errorG.item(),D_x,D_G_z1,D_G_z2))
        G_losses.append(errorG)
        D_losses.append(errorD)
        if i%500==0 or (e==epochs-1 and i==len(dataloader)-1):
            with torch.no_grad():
                fake=netG(noise).detach().cpu()
                img_list.append(vutils.make_grid(fake,padding=2,normalize=True))
        iters+=1
            
torch.save({
        "Dmodel":netD.state_dict(),
        "Gmodel":netG.state_dict()
        },"dcgan.pth")
            
        

        
        