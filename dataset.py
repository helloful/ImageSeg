import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random
import pickle
from PIL import Image

def transform1(image, gt,crop_size=256):
    '''

    :param image:       PIL图像patch
    :param gt:          PIL图像的groundtruth
    :return:            统一的随机处理增强以后的Torch.tensor

    '''
    ##################################################
    # 配置
    CJ_PROB                     = 0.1
    CJ_BRITNESS                 = 0.02
    CJ_CONTRAST                 = 0.02
    CJ_SATURATION               = 0.02
    CJ_HUE                      = 0.3
    RC_SIZE                     = 256
    CC_PROB                     = 0.3
    ##################################################
    # Resize
    #resize = transforms.Resize(size=(RC_SIZE, RC_SIZE))
    #image = resize(image)
    #mask = resize(mask)

    # Random crop
 
    try:
        i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(crop_size,crop_size))
        image = TF.crop(image, i, j, h, w)
        gt = TF.crop(gt, i, j, h, w)
    except Exception as  e:
        print("trans:",e)


    # random rotation
    if random.random() > 0.5:
        image = image.transpose(Image.ROTATE_90)
        gt = gt.transpose(Image.ROTATE_90)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        gt = TF.hflip(gt)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        gt = TF.vflip(gt)

    # Color Jittering,改变亮度，对比度，饱和度
    # if random.random() < CJ_PROB:
    #     colorTrans = transforms.ColorJitter.get_params((1-CJ_BRITNESS,1+CJ_BRITNESS), (1-CJ_CONTRAST,1+CJ_CONTRAST), (1-CJ_SATURATION,1+CJ_SATURATION), (0-CJ_HUE,CJ_HUE))
    #     # colorTrans=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    #     image = colorTrans(image)
    #     gt = colorTrans(gt)


    # Transform to tensor
    image = TF.to_tensor(image)
    gt = TF.to_tensor(gt)

    # 随机打乱通道顺序
    if random.random() < CC_PROB :
        idx = torch.randperm(3)
        image = image[idx,:,:]
        gt    = gt[idx,:,:]

    return image, gt

class MyDataset(Dataset):
    def __init__(self,data_path,is_train=1,RC_SIZE=256  ):
        super(MyDataset,self).__init__()
        self.is_train=is_train
        with open(data_path,"rb") as f:
            self.data=pickle.load(f)
        if self.is_train==1:
            self.length=len(self.data['train_img'])
        elif self.is_train==2:
            self.length=len(self.data['test_img'])
        # self.trans=TF.to_tensor
        self.trans=transforms.Compose([
            transforms.Resize(size=(RC_SIZE, RC_SIZE)),
            TF.to_tensor,
            transforms.Normalize([0.5,0.5,0.5,],[0.5,0.5,0.5])
        ])
        self.trans2=transforms.Compose([
            TF.to_tensor,
        ])
        
    def __getitem__(self, index):
        if self.is_train==1:
            img=Image.open(self.data['train_img'][index]).convert('RGB')
            mask=Image.open(self.data['train_mask'][index]).convert("1")
            return self.trans(img),self.trans2(mask)
        elif self.is_train==2:
            img=Image.open(self.data['test_img'][index]).convert('RGB')
            mask=Image.open(self.data['test_mask'][index]).convert('1')
            return self.trans(img),self.trans2(mask)        

    def __len__(self):
        return self.length