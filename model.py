import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,in_channel=3,out_channel=1):
        super(Model,self).__init__()
        self.pooling=nn.MaxPool2d((2,2))
        self.upsample=nn.Upsample(scale_factor=2,mode='bicubic')
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channel,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(inplace=True),
        )
        #concat[layer4_out,layer3_out]
        self.layer5=nn.Sequential(
            nn.Conv2d(512,128,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(inplace=True),
        )
        #concat[layer5_out,layer3_out]
        self.layer6=nn.Sequential(
            nn.Conv2d(256,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
        )
        #concat[layer5_out,layer3_out]
        self.layer7=nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x1=self.layer1(x)
        x1_pool=self.pooling(x1)

        x2=self.layer2(x1_pool)
        x2_pool=self.pooling(x2)

        x3=self.layer3(x2_pool)
        x3_pool=self.pooling(x3)

        x4=self.layer4(x3_pool)
        x4_pool=self.upsample(x4)

        x5=self.layer5(torch.cat([x4_pool,x3],dim=1))
        x5_pool=self.upsample(x5)

        x6=self.layer6(torch.cat([x5_pool,x2],dim=1))
        x6_pool=self.upsample(x6)

        x7=self.layer7(torch.cat([x6_pool,x1],dim=1))

        return x7


def main():
    model=Model().cuda()
    a=torch.randn((8,3,256,256)).cuda()
    y=model(a)
    print(a.shape,y.shape)
    

if __name__ == '__main__':
    main()
    

        

