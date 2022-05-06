
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



        
def valid(model=None,test_dataloader=None,data_name='COVID'):
    print("Validing")
    if model is None:
        model=Model().cuda()
        checkpoint=torch.load("./exp/model2_epoch_190.pth")
        model.load_state_dict(checkpoint['state_dict'])
        # 初始化数据集
    if test_dataloader is None:
        test_data=MyDataset("/home/user1/homework/yxyxfx/code/{}_dataset.pkl".format(data_name),is_train=2)
        test_dataloader=DataLoader(test_data,batch_size=8,num_workers=8)

    with torch.no_grad():
        model.eval()
        total_iou=0.0
        total_dice=0.0
        total=0.0
        for iter,(img,mask) in enumerate(test_dataloader):
            img=img.cuda()
            mask=mask.cuda()
            out = model(img)
            res = out.detach().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res_img=Image.fromarray(res[0]*255)
            res_img.convert('RGB').save("valid.png")
            total_iou+=iou(out,mask)*mask.shape[0]
            total_dice+=dice_coef(out,mask)*mask.shape[0]
            total+=mask.shape[0]
        print("Valid Avg iou: {}".format(total_iou/total))
        print("Valid Avg dice: {}".format(total_dice/total))



if __name__ == '__main__':

    valid(data_name='Lung_Opacity')
  