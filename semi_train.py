import shutil
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from semi_dataset import train_dataset,test_dataset
from model2 import Model
from dataset import MyDataset
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
EPOCH=100
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

def train_stage1():
    # 初始化模型
    init()
    model=Model().cuda()
    # model=Model().cuda()
    # model.load_state_dict(torch.load("./exp/model_epoch_155.pth")['state_dict'])

    # 初始化数据集
    train_data=train_dataset("/home/user1/homework/yxyxfx/code/exp_semi/imgs/","/home/user1/homework/yxyxfx/code/exp_semi/masks/")
    train_dataloader=DataLoader(train_data,batch_size=8,num_workers=8)
   
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
            if iter %10==0 and iter>0:
                print("Iter:{} loss: {}".format(iter,loss.item()))
                print("Train Epoch: {}, avg iou: {}, avg dice: {}".format(epoch,total_iou/total,total_dice/total))
                wandb.log({"train_loss":total_loss/iter,"train avg iou":total_iou/total,"train avg dice":total_dice/total})
        # torch.save(model.state_dict(),"model2_{}.pth".format(epoch))
        print("Adjust Learning rate: {}".format(scheduler.get_lr()))
        wandb.log({"lr":scheduler.get_lr()[0]})
        scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if epoch>0 and epoch%10==0:
            torch.save(checkpoint,"./exp_semi/stage1/model_epoch_{}.pth".format(epoch))

def valid_stage1(pth='./exp_semi/stage1/model_epoch_90.pth'):
    # 把通过第一阶段的训练的模型加载生成伪标签
    model=Model().cuda()
    model.load_state_dict(torch.load(pth)['state_dict'])
    model.eval()
    valid_path='/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/images/'
    save_mask='/home/user1/homework/yxyxfx/code/exp_semi/stage2/mask/'

    test_loader=test_dataset(valid_path)

    for i in range(test_loader.length):
        img,name=test_loader.load_data()
        img=img.cuda()
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(img)

        res = lateral_map_2.sigmoid().detach().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res_img=Image.fromarray(res*255)
        res_img.convert('RGB').save(os.path.join(save_mask,name))
    # 把真标签复制过来替换掉
    gt_path='/home/user1/homework/yxyxfx/code/exp_semi/masks'
    gt_mask=os.listdir(gt_path)
    for item in gt_mask:
        from_mask=os.path.join(gt_path,item)
        to_mask=os.path.join(save_mask,item)
        shutil.copyfile(from_mask,to_mask)



def train_model(pth_path,img_path,gt_path):
    # 初始化模型
    # init()
    model=Model().cuda()
    # model=Model().cuda()
    model.load_state_dict(torch.load(pth_path)['state_dict'])

    # 初始化数据集
    train_data=train_dataset(img_path,gt_path)
    train_dataloader=DataLoader(train_data,batch_size=8,num_workers=8)
   
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
            if iter %10==0 and iter>0:
                print("Iter:{} loss: {}".format(iter,loss.item()))
                print("Train Epoch: {}, avg iou: {}, avg dice: {}".format(epoch,total_iou/total,total_dice/total))
                wandb.log({"train_loss":total_loss/iter,"train avg iou":total_iou/total,"train avg dice":total_dice/total})
        # torch.save(model.state_dict(),"model2_{}.pth".format(epoch))
        print("Adjust Learning rate: {}".format(scheduler.get_lr()))
        wandb.log({"lr":scheduler.get_lr()[0]})
        scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if epoch>0 and epoch%10==0:
            torch.save(checkpoint,"./exp_semi/stage1/model_epoch_{}.pth".format(epoch))

def train():
    print("traing stage1.....")
    train_stage1()
    print("valid stage1.....")
    valid_stage1()
    # 迭代多次
    for stage in range(10):
        #######
        # 1、用当前的真标签和伪标签训练
        ######
        print("traing stage{}.....".format(stage))
        train_model('/home/user1/homework/yxyxfx/code/exp_semi/stage1/model_epoch_90.pth',
            "/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/images/",
            '/home/user1/homework/yxyxfx/code/exp_semi/stage2/mask/'
        )
        valid_stage1("/home/user1/homework/yxyxfx/code/exp_semi/stage1/model_epoch_90.pth")


def valid(model=None,test_dataloader=None,dataset='COVID',train=False):
  
    if model is None:
        model=Model().cuda()
        checkpoint=torch.load("/home/user1/homework/yxyxfx/code/exp_semi/stage1/50_model_epoch_90.pth")
        model.load_state_dict(checkpoint['state_dict'])
        # 初始化数据集
    if test_dataloader is None:
        test_data=MyDataset("/home/user1/homework/yxyxfx/code/{}_dataset.pkl".format(dataset),is_train=2)
        test_dataloader=DataLoader(test_data,batch_size=8,num_workers=8)
    with torch.no_grad():
        model.eval()
        total_iou=0.0
        total_dice=0.0
        total=0.0
        for iter,(img,mask) in enumerate(test_dataloader):
            img=img.cuda()
            mask=mask.cuda()
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(img)
            
            #包图像保存到wandb
            if iter==10 and train:
                w_img=img[0].permute(1,2,0).detach().cpu().numpy()
                w_img=np.array(w_img*255)
                w_img=wandb.Image(w_img,caption="Lung Image")
                w_mask=lateral_map_2.sigmoid().detach().cpu().numpy().squeeze()
                w_mask=w_mask[0]
                w_mask[w_mask>0.5]=1
                w_mask[w_mask<=0.5]=0
                
                class_labels = {
                0: "chest",
                1: "lung",
                }
                mask_img = wandb.Image(w_img, masks={
                "predictions": {
                    "mask_data": w_mask,
                    "class_labels": class_labels
                }
                })
                mask_img2=wandb.Image(w_mask*255,caption="mask Image")
                wandb.log({"img_sample": mask_img,"mask":mask_img2})



            # 图像的本地保存
            if   train:
                res = lateral_map_2.sigmoid().detach().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res_img=Image.fromarray(res*255)
                res_img.convert('RGB').save("valid.png")

            # 计算分割指标
            total_iou+=iou(lateral_map_2.sigmoid(),mask)*mask.shape[0]
            total_dice+=dice_coef(lateral_map_2.sigmoid(),mask)*mask.shape[0]
            total+=mask.shape[0]
        print("Valid Avg iou: {}".format(total_iou/total))
        print("Valid Avg dice: {}".format(total_dice/total))
        if train:
            wandb.log({"valid avg iou":total_iou/total,"valid avg dice":total_dice/total})
        
        

        

          


if __name__ == '__main__':
    # train_stage1()
    # train()
    valid(dataset='Lung_Opacity')
    valid(dataset='COVID')
    valid(dataset='Normal')
    valid(dataset='Viral Pneumonia')
