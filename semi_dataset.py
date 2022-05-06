import os
import torch
import torch.nn as nn
import numpy as np
import random
import shutil
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset
def move_data():
    img_path="/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/images"
    mask_path="/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/masks"
    save_img='/home/user1/homework/yxyxfx/code/exp_semi/imgs'
    save_mask='/home/user1/homework/yxyxfx/code/exp_semi/masks'

    # 任意选择50张作为GT
    img_list=os.listdir(img_path)

    sample_list=list(np.random.choice(len(img_list),100))
    for i in sample_list:
        from_img=os.path.join(img_path,img_list[i])
        to_img=os.path.join(save_img,img_list[i])
        shutil.copyfile(from_img,to_img)
        from_mask=os.path.join(mask_path,img_list[i])
        to_mask=os.path.join(save_mask,img_list[i])
        shutil.copyfile(from_mask,to_mask)
class test_dataset:
    def __init__(self,img_path,img_size=256) :
        self.img_size=img_size
        self.img_list=[img_path+f for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith(".png")]
        self.trans=transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     [0.485, 0.456, 0.406],
            #      [0.229, 0.224, 0.225]
            # )
        ])
        self.length=len(self.img_list)
        self.index=0
    def load_data(self):
        imgs=self.read_img(self.img_list[self.index])
        imgs=self.trans(imgs).unsqueeze(0)
        name=self.img_list[self.index].split("/")[-1]
        if name.endswith(".jpg"):
            name=name.split(".jpg")[0]+".png"
        self.index+=1
        return imgs,name


    def read_img(self,path):
        img=Image.open(path).convert("RGB")
        return img
class train_dataset(Dataset):
    def __init__(self,img_path,gt_path,img_size=256):
        self.img_size=img_size
        self.img_list=[img_path+f for f in os.listdir(img_path) if f.endswith(".jpg") or f.endswith(".png")]
        self.gt_list=[gt_path+f for f in os.listdir(gt_path) if f.endswith(".jpg") or f.endswith(".png")]
        self.img_list=sorted(self.img_list)
        self.gt_list=sorted(self.gt_list)

        self.length=len(self.gt_list)
        self.img_trans=transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        imgs=Image.open(self.img_list[index]).convert("RGB")
        gt=Image.open(self.gt_list[index]).convert("1")
        imgs=self.img_trans(imgs)
        gts=self.img_trans(gt)
        return imgs,gts
    def __len__(self):
        return self.length




if __name__ == '__main__':
    move_data()
    