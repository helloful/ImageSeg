
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


def valid(model=None,test_dataloader=None,dataset='COVID',train=False):
    print("Validing")
    if model is None:
        model=Model().cuda()
        checkpoint=torch.load("./exp2/model_epoch_195.pth")
        model.load_state_dict(checkpoint['state_dict'])
        # 初始化数据集
    if test_dataloader is None:
        test_data=MyDataset("/home/user1/homework/yxyxfx/code/{}_dataset.pkl".format(dataset),is_train=2)
        test_dataloader=DataLoader(test_data,batch_size=8,num_workers=8)
    sava_path='/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/'+dataset
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
            if not train:
                res = lateral_map_2.sigmoid().detach().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res_img=Image.fromarray(res[0]*255)
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
     valid(dataset='Lung_Opacity')
     valid(dataset='COVID')
     valid(dataset='Normal')
     valid(dataset='Viral Pneumonia')