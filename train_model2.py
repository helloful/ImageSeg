
from tabnanny import check
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import pickle
from dataset import MyDataset
from model2 import Model
# from model2 import Model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from metrics import iou,dice_coef
import random
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb

# 这个训练
EPOCH=200
seed=321
# 设置随机种子初始化
# 设置随机种子额初始化
def init():
    wandb.init(project="covid_lung_segmentation", entity="hwlloful")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

learning_rate=2e-4

wandb.config={
    "learning_rate":learning_rate,
    "epochs":EPOCH,
    "batch_size":8
}
def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train():
    # 初始化模型
    init()
    model=Model().cuda()
    # model=Model().cuda()
    # model.load_state_dict(torch.load("./exp/model_epoch_155.pth")['state_dict'])

    # 初始化数据集
    train_data=MyDataset("/home/user1/homework/yxyxfx/code/COVID_dataset.pkl",is_train=1)
    train_dataloader=DataLoader(train_data,batch_size=8,num_workers=8)
    test_data=MyDataset("/home/user1/homework/yxyxfx/code/COVID_dataset.pkl",is_train=2)
    test_dataloader=DataLoader(test_data,batch_size=8,num_workers=8)
    # 定义优化模型和损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-8)
    
    for epoch in range(EPOCH):
        model.train()
        total_iou=0.0
        total_dice=0.0
        total=0.0
        total_loss=0
        for iter,(img,mask) in enumerate(train_dataloader):
            img=img.cuda()
            mask=mask.cuda()
            # out=model(img)
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(img)
            loss5 = joint_loss(lateral_map_5, mask)
            loss4 = joint_loss(lateral_map_4, mask)
            loss3 = joint_loss(lateral_map_3, mask)
            loss2 = joint_loss(lateral_map_2, mask)
            loss = loss2 + loss3 + loss4 + loss5
            total_loss+=loss.item()
            ####################
            # 计算IOU，Dice [batch_size,1,256,256]
            # 下面这个计算的时候，如果不加with.no_grad胡存在内存爆炸的问题
            # 目前还不理解为什么会这样
            ####################
            with torch.no_grad():
                total_iou+=iou(lateral_map_2.sigmoid(),mask)*mask.shape[0]
                total_dice+=dice_coef(lateral_map_2.sigmoid(),mask)*mask.shape[0]
                total+=mask.shape[0]
           
            ###---- backward ----  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter %5==0 and iter>0:
                print("Iter:{} loss: {}".format(iter,loss.item()))
                print("Train Epoch: {}, avg iou: {}, avg dice: {}".format(epoch,total_iou/total,total_dice/total))
                wandb.log({"train_loss":total_loss/iter,"train avg iou":total_iou/total,"train avg dice":total_dice/total})
        # torch.save(model.state_dict(),"model2_{}.pth".format(epoch))
        # valid(model,test_dataloader,train=True)
        print("Adjust Learning rate: {}".format(scheduler.get_lr()))
        wandb.log({"lr":scheduler.get_last_lr()[0]})
        scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if epoch>0 and epoch%5==0:
            torch.save(checkpoint,"./exp2/model_epoch_{}.pth".format(epoch))




        

def test():
    # create_dataset("COVID")
    print("测试")
    data=MyDataset('/home/user1/homework/yxyxfx/code/COVID_dataset.pkl',is_train=1)
    dataloader=DataLoader(data,num_workers=1,batch_size=8)
    for iter,(img,mask) in enumerate(dataloader):
        print(iter,img.shape,mask.shape,torch.max(img),torch.min(img),torch.max(mask),torch.min(mask))
        break
      


if __name__ == '__main__':
    train()