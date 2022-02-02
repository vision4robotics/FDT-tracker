import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.utile.tran import Transformer

    
class hiftmodule(nn.Module):
    
    def __init__(self,cfg):
        super(hiftmodule, self).__init__()
        self.lastres = t.zeros(1,1,1,1).cuda()
        self.num=0
        
        channel=192
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3, bias=False, stride=2,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            )
        
        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),                
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        
        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.row_embed = nn.Embedding(50, 192//2)
        self.col_embed = nn.Embedding(50, 192//2)
        self.reset_parameters()
        
        self.trans = Transformer(192, 6,1,1)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)
        for modules in [self.convloc, self.convcls,
                        self.cls1, self.cls2]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        for modules in [self.conv3]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
        
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        
    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):
        
        res=self.xcorr_depthwise(x, z)
        
        res=self.conv3(res)
        
        h2, w2 = 11, 11
        i = t.arange(w2).cuda()
        j = t.arange(h2).cuda()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = t.cat([
            x_emb.unsqueeze(0).repeat(h2, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w2, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(z.shape[0], 1, 1, 1)

        b, c, w, h=z.size()
        
       
        if self.num==0 :
           resx = res
           self.num=self.num+1
        elif self.num == 1:
         ress = self.lastres.reshape(b,192,11,11)
         resx=self.trans((pos+res).view(b,192,-1).permute(2, 0, 1),\
                          (pos+ress).view(b,192,-1).permute(2, 0, 1))
         
         resx=resx.permute(1,2,0).view(b,192,11,11)
        guider=resx.reshape(b*192*11*11)
        self.lastres = guider.clone().detach()
        loc=self.convloc(resx)
        acls=self.convcls(resx)

        cls1=self.cls1(acls)
        cls2=self.cls2(acls)
        
        return loc,cls1,cls2





