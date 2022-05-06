
from tabnanny import check
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import pickle
from dataset import MyDataset
from model import Model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from metrics import iou,dice_coef
import random
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCH=500
seed=321
# 设置随机种子初始化
# 设置随机种子额初始化
def init():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

learning_rate=2e-4
def create_dataset(data_name,factor=0.8):
    data_path='/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset'
    data_path=os.path.join(data_path,data_name)
    imgs_path=os.path.join(data_path,'images')
    mask_path=os.path.join(data_path,'masks')
    imgs_list=sorted(os.listdir(imgs_path))
    masks_list=sorted(os.listdir(mask_path))
    print(len(imgs_list),len(masks_list))
    path1=[]
    path2=[]
    for img in imgs_list:
        path1.append(os.path.join(imgs_path,img))
        path2.append(os.path.join(mask_path,img)) 
    imgs_list=path1
    masks_list=path2

    total=len(imgs_list)
    train_img_set=imgs_list[:int(total*factor)]
    train_mask_set=masks_list[:int(total*factor)]
    test_img_set=imgs_list[int(total*factor):]
    test_mask_set=masks_list[int(total*factor):]
    
    data={}
    data['train_img']=train_img_set
    data['train_mask']=train_mask_set
    data['test_img']=test_img_set
    data['test_mask']=test_mask_set
    # with open("{}_dataset.pkl".format(data_name),"wb") as f:
    pickle.dump(data,open('{}_dataset.pkl'.format(data_name), 'wb'))




def train():
    # 初始化模型
    init()
    # model=Inf_Net().cuda()
    model=Model().cuda()
    # checkpoint=torch.load("./exp/model2_epoch_192.pth")
    # model.load_state_dict(checkpoint['state_dict'])

    # 初始化数据集
    train_data=MyDataset("/home/user1/homework/yxyxfx/code/COVID_dataset.pkl",is_train=1)
    train_dataloader=DataLoader(train_data,batch_size=8,num_workers=8)
    test_data=MyDataset("/home/user1/homework/yxyxfx/code/COVID_dataset.pkl",is_train=2)
    test_dataloader=DataLoader(test_data,batch_size=8,num_workers=8)


    # 定义优化模型和损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-8)
    loss_func=nn.BCEWithLogitsLoss()
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    loss_func=F.binary_cross_entropy
    conti=checkpoint['epoch']
    for epoch in range(conti,EPOCH):
        model.train()
        total_iou=0.0
        total_dice=0.0
        total=0.0
        for iter,(img,mask) in enumerate(train_dataloader):
            img=img.cuda()
            mask=mask.cuda()
            out=model(img)
            loss=loss_func(out,mask)
          
           
            ####################
            # 计算IOU，Dice [batch_size,1,256,256]
            # 下面这个计算的时候，如果不加with.no_grad胡存在内存爆炸的问题
            # 目前还不理解为什么会这样
            ####################
            with torch.no_grad():
                total_iou+=iou(out,mask)*mask.shape[0]
                total_dice+=dice_coef(out,mask)*mask.shape[0]
                total+=mask.shape[0]
           
            ###---- backward ----  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter %100==0:
                print("Iter:{} loss: {}".format(iter,loss.item()))
                print("Train Epoch: {}, avg iou: {}, avg dice: {}".format(epoch,total_iou/total,total_dice/total))
        # torch.save(model.state_dict(),"model2_{}.pth".format(epoch))
        # valid2(model,test_dataloader)
        print("Adjust Learning rate: {}".format(scheduler.get_last_lr()))
        scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint,"./exp/model2_epoch_{}.pth".format(epoch))


      


if __name__ == '__main__':
    train()
   