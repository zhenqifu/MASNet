import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.Res2Net_v1b import res2net50_v1b_26w_4s


def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Outlayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Outlayer, self).__init__()
        self.out = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x = self.out(x)
        return x


class Simam(nn.Module):
    def __init__(self, e=1e-4):
        super(Simam, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.e = e

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h -1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e)) + 0.5

        return self.sigmoid(y)


class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.simam = Simam()

    def forward(self, x): 
        out = self.simam(x)
        return out * x


class Fusion_sum(nn.Module):
    def __init__(self):
        super(Fusion_sum, self).__init__()
        self.attention = attention()
        self.upsample = cus_sample

    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        xy = x + y
        out = self.attention(xy)
        return out


class Fusion_sum2(nn.Module):
    def __init__(self):
        super(Fusion_sum2, self).__init__()
        self.attention = attention()
        self.upsample = cus_sample

    def forward(self, x, y):
        xy = x + y
        out = self.attention(xy)
        return out


class Net(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.rfb0 = RFB_modified(64, 64)
        self.rfb1 = RFB_modified(256, 64)        
        self.rfb2 = RFB_modified(512, 64)
        self.rfb3 = RFB_modified(1024, 64)
        self.rfb4 = RFB_modified(2048, 64)

        self.fusion4_sum = Fusion_sum()
        self.fusion3_sum = Fusion_sum()
        self.fusion2_sum = Fusion_sum()
        self.fusion1_sum = Fusion_sum2()        
        self.classifier = nn.Conv2d(64, 1, 1)

    def forward(self, x):
#================= Backbone ==============================
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)   # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)   # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

#================= RFB ==============================

        x0_rfb = self.rfb0(x)  
        x1_rfb = self.rfb1(x1)  
        x2_rfb = self.rfb2(x2)  
        x3_rfb = self.rfb3(x3) 
        x4_rfb = self.rfb4(x4) 
        
#================= Fusion ==============================

        out43_sum = self.fusion4_sum(x3_rfb, x4_rfb)
        out432_sum = self.fusion3_sum(x2_rfb, out43_sum)
        out10_sum = self.fusion1_sum(x0_rfb, x1_rfb)
        out43210_sum = self.fusion2_sum(out10_sum, out432_sum)

        s3 = self.classifier(out43210_sum)
        s3 = F.interpolate(s3, scale_factor=4, mode='bilinear', align_corners=False)

        return F.sigmoid(s3)
