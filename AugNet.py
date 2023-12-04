import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from tools.padding_same_conv import Conv2d




class Group1(nn.Module):
    def __init__(self,):
        super(Group1, self).__init__()
        self.conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(16)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group2(nn.Module):
    def __init__(self,):
        super(Group2, self).__init__()
        self.conv = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group3(nn.Module):
    def __init__(self,):
        super(Group3, self).__init__()
        self.conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(64)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group4(nn.Module):
    def __init__(self,):
        super(Group4, self).__init__()
        self.conv = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out


class Group5(nn.Module):
    def __init__(self,):
        super(Group5, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group6(nn.Module):
    def __init__(self,):
        super(Group6, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group7(nn.Module):
    def __init__(self,):
        super(Group7, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group8(nn.Module):
    def __init__(self,):
        super(Group8, self).__init__()
        self.conv = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.LeakyReLU(negative_slope=0.2, inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group9(nn.Module):
    def __init__(self,):
        super(Group9, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group10(nn.Module):
    def __init__(self,):
        super(Group10, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group11(nn.Module):
    def __init__(self,):
        super(Group11, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group12(nn.Module):
    def __init__(self,):
        super(Group12, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group13(nn.Module):
    def __init__(self,):
        super(Group13, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group14(nn.Module):
    def __init__(self,):
        super(Group14, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group15(nn.Module):
    def __init__(self,):
        super(Group15, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.actv = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.actv(out)
        return out

class Group16(nn.Module):
    def __init__(self,):
        super(Group16, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Group17(nn.Module):
    def __init__(self,):
        super(Group17, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.sigmoid(x)-0.5
        out = self.ReLU(out)
        return out



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.block1 = Group1()
        self.block2 = Group2()
        self.block3 = Group3()
        self.block4 = Group4()
        self.block5 = Group5()
        self.block6 = Group6()
        self.block7 = Group7()
        self.block8 = Group8()
        self.block9 = Group9()
        self.block10 = Group10()
        self.block11 = Group11()
        self.block12 = Group12()
        self.block13 = Group13()
        self.block14 = Group14()
        self.block15 = Group15()
        self.block16 = Group16()
        self.block17 = Group17()
        

    def forward(self, x):
        x = self.block1(x)
        output1 = x

        x = self.block2(x)
        output2 = x

        x = self.block3(x)
        output3 = x
  
        x = self.block4(x)
        output4 = x

        x = self.block5(x)
        output5 = x
 
        x = self.block6(x)
        output6 = x

        x = self.block7(x)
        output7 = x

        x = self.block8(x)

        x = self.block9(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x,output7],dim=1)

        x = self.block10(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x,output6],dim=1)

        x = self.block11(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.cat([x,output5],dim=1)
 
        x = self.block12(x)
        x = torch.cat([x,output4],dim=1)

        x = self.block13(x)
        x = torch.cat([x,output3],dim=1)

        x = self.block14(x)
        x = torch.cat([x,output2],dim=1)

        x = self.block15(x)
        x = torch.cat([x,output1],dim=1)

        x = self.block16(x)

        x = self.block17(x)
        
        return x


if __name__=='__main__':
    x = torch.randn(8,1,256,256).cuda()
    model = UNet().cuda()
    y = model(x)

