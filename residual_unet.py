import torch
import torch.nn as nn
from  torchvision import models 
from torch.nn.functional import sigmoid as sigmoid
from torch.autograd import Variable
'''
Double_conv: series of 2 3x3 conv
Note: In the unet paper no padding is added. To preseve the spatial dimensions, padding = 1 is included
'''
#Gaussian Dropout is taken from https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
#note that alpha = p/1-p, alpha = 1, means p = 0.5
class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x

def double_conv(in_channels, out_channels):
    
            return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )   


class upblock(nn.Module):
    
    def __init__(self, in_channels, out_channels, merge_type = "concat"):
        super(upblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_type = merge_type
        self.conv = double_conv(in_channels, out_channels)
        if(merge_type == "concat"):
            # Divide out_channels by 2, will be merged with the preceding down_blocks
            self.upconv = nn.ConvTranspose2d(out_channels,out_channels // 2,kernel_size=2,stride=2) 
        elif(merge_type == "add"):
            self.upconv = nn.ConvTranspose2d(out_channels,out_channels ,kernel_size=2,stride=2) 

    def forward(self, x):
        # Two convolutions followed by upconvolution
        # Chosen upconvolution is Transpose Convolution
        x = self.conv(x) 
        if(self.merge_type == "concat"):
            x = self.upconv(x)
        elif(self.merge_type == "add"):
            x = self.upconv(x)
        return x


class res_unet(nn.Module):
    
    def __init__(self, num_classes = 2,in_channels = 3, dropout = 0,drop_type = "normal"):
        super(res_unet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.drop = nn.Dropout2d()
        self.preprocess = nn.Sequential(*list(resnet.children()))[:4] #64, 128, 256
        self.down1 = nn.Sequential(*list(resnet.children()))[4] # 64, 128, 256
        self.down2 = nn.Sequential(*list(resnet.children()))[5] #128, 64, 128
        self.down3 = nn.Sequential(*list(resnet.children()))[6] # 256, 32, 64

        self.up1 = upblock(in_channels = 256, out_channels = 256) #Returns out_channels / 2 , and spatial size x 2
        self.up2 = upblock(in_channels = 256, out_channels = 128)
        self.up3 = nn.Conv2d(128,64,1) # Note down1 and preprocess has the same spatial dimension 
        self.up4 = upblock(in_channels = 128, out_channels = 256)
        self.up5 = upblock(in_channels = 128, out_channels=  128)
        self.logits = nn.Conv2d(64, num_classes, kernel_size=1)
        if(drop_type == "normal"):
            self.drop = nn.Dropout2d()
        elif(drop_type == "gaussian"):
            self.drop = GaussianDropout()
        
        

    def forward(self,x):

        base = self.preprocess(x)
        down1 = self.down1(base)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        up = torch.cat([self.up1(down3),down2],dim = 1) #256 x 64 x 128
        if(self.dropout ==3 or self.dropout == "all"):
            up = self.drop(up)
        up = torch.cat([self.up2(up),down1],dim = 1)  #128 x 128 x 256
        if(self.dropout == 2 or self.dropout== "all"):
            up = self.drop(up)
        up = torch.cat([self.up3(up),base],dim = 1)   #128 x 128 x 256
        if(self.dropout == 1 or self.dropout=="all"):
            up = self.drop(up)
        up = self.up5(self.up4(up)) #64 x 512 x 1024
        logits_layer = self.logits(up)
        return logits_layer



class res_unet_add(nn.Module):
    
    def __init__(self, num_classes = 2,in_channels = 3, dropout = 0):
        super(res_unet_add, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        self.drop = nn.Dropout2d()
        self.preprocess = nn.Sequential(*list(resnet.children()))[:4] #64, 128, 256
        self.down1 = nn.Sequential(*list(resnet.children()))[4] # 64, 128, 256
        self.down2 = nn.Sequential(*list(resnet.children()))[5] #128, 64, 128
        self.down3 = nn.Sequential(*list(resnet.children()))[6] # 256, 32, 64

        self.up1 = upblock(in_channels = 256, out_channels = 128,merge_type = "add") #128x64x128
        self.up2 = upblock(in_channels = 128, out_channels = 64,merge_type = "add") #64x128x256
        self.up4 = upblock(in_channels = 64, out_channels = 64,merge_type="add")
        self.up5 = upblock(in_channels = 64, out_channels=  64,merge_type="add")
        self.logits = nn.Conv2d(64, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d()
        
        

    def forward(self,x):

        base = self.preprocess(x)
        down1 = self.down1(base)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        up = self.up1(down3)+down2 #128 x 64 x 128
        up = self.up2(up)+down1  #64 x 128 x 256
        up = up+base  #64 x 128 x 256
        up = self.up5(self.up4(up)) #64 x 512 x 1024
        logits_layer = self.logits(up)
        return logits_layer




class gate(nn.Module):
    def __init__(self, in_channels):
        super(gate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels, kernel_size = (1,1))
        self.conv2 = nn.Conv2d(in_channels,in_channels, kernel_size = (1,1))
        self.in_channels = in_channels

    def forward(self, e,d):
        #d is the decoder at layer i
        #e is the encoder at layer i
        alpha = torch.sigmoid(self.conv1(d))
        beta = torch.sigmoid(self.conv2(e))
        return (1+alpha)*d + (1-alpha)*(beta*e)
    
class res_unet_gate(nn.Module):
    
    def __init__(self, num_classes = 2,in_channels = 3, dropout = 0,drop_type = "gaussian"):
        super(res_unet_gate, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout
        if(drop_type == "normal"):
            self.drop = nn.Dropout2d()
        elif(drop_type == "gaussian"):
            self.drop = GaussianDropout(alpha=1)
        
        
        
        self.preprocess = nn.Sequential(*list(resnet.children()))[:4] #64, 128, 256
        self.down1 = nn.Sequential(*list(resnet.children()))[4] # 64, 128, 256
        self.down2 = nn.Sequential(*list(resnet.children()))[5] #128, 64, 128
        self.down3 = nn.Sequential(*list(resnet.children()))[6] # 256, 32, 64

        self.up1 = upblock(in_channels = 256, out_channels = 128,merge_type = "add") #128x64x128
        self.up2 = upblock(in_channels = 128, out_channels = 64,merge_type = "add") #64x128x256
        self.up3 = upblock(in_channels = 64, out_channels = 64,merge_type="add")

        self.gate1 = gate(128)
        self.gate2 = gate(64)
        self.gate3 = gate(64)
        self.up4 = upblock(in_channels = 64, out_channels=  64,merge_type="add")
        self.logits = nn.Conv2d(64, num_classes, kernel_size=1)
        
        

    def forward(self,x):

        base = self.preprocess(x)
        down1 = self.down1(base)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        up = self.gate1(down2,self.up1(down3)) #128 x 64 x 128
        if(self.dropout == 1 or self.dropout=="all"):
            up = self.drop(up)
       
        up = self.gate2(down1,self.up2(up))  #64 x 128 x 256
        if(self.dropout=="all"):
            up = self.drop(up)
        up = self.gate3(base,up) #64 x 128 x 256
        if(self.dropout=="all"):
            up=self.drop(up)
        up = self.up4(self.up3(up)) #64 x 512 x 1024
        logits_layer = self.logits(up)
        return logits_layer







