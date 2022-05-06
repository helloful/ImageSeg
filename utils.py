from operator import imod
import os
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import shutil
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

# CV2读取的图像格式为HWC
def add_colored_dilate(image, mask_image, dilate_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    dilate_image_gray = cv2.cvtColor(dilate_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    dilate = cv2.bitwise_and(dilate_image, dilate_image, mask=dilate_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])
    dilate_coord = np.where(dilate!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]
    dilate[dilate_coord[0],dilate_coord[1],:] = [0,0,255]

    ret = cv2.addWeighted(image, 0.7, dilate, 0.3, 0)
    ret = cv2.addWeighted(ret, 0.7, mask, 0.3, 0)

    return ret

def add_colored_mask(image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return ret

def diff_mask(ref_image, mask_image):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    
    mask_coord = np.where(mask!=[0,0,0])

    mask[mask_coord[0],mask_coord[1],:]=[255,0,0]

    ret = cv2.addWeighted(ref_image, 0.7, mask, 0.3, 0)
    return ret
def showImage(img_name):
    
    img_path='/home/user1/homework/yxyxfx/code/test/{}-img.png'.format(img_name)
    mask_path='/home/user1/homework/yxyxfx/code/test/{}-mask.png'.format(img_name)
    pred_path='/home/user1/homework/yxyxfx/code/test/{}-premask.png'.format(img_name)

    img=cv2.imread(img_path)
    mask=cv2.imread(mask_path)
    predict_mask=cv2.imread(pred_path)
    rgb_image = cv2.resize(img,(256, 256))
    merged_image = add_colored_dilate(rgb_image, mask, rgb_image)
    merged_img2=add_colored_dilate(rgb_image,predict_mask,rgb_image)
                          
    print(np.max(img),np.min(img))
    print(np.max(mask),np.min(mask))

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title(img_name)
    plt.subplot(2,3,2)
    plt.imshow(mask)
    plt.title("GT mask")
    plt.subplot(2,3,3)
    plt.imshow(merged_image)
    plt.title("Merge Image")

    plt.subplot(2,3,4)
    plt.imshow(img)
    plt.title(img_name)
    plt.subplot(2,3,5)
    plt.imshow(predict_mask)
    plt.title("Pred Mask")
    # mask=cv2.imread(mask_path,0)
    # print(np.histogram(np.array(mask[:,:,0])))

    plt.subplot(2,3,6)
    plt.imshow(merged_img2)
    plt.title("Merge Image")



    plt.show()
    plt.savefig("plt1_{}.png".format(img_name))
def valid3(model=None,test_dataloader=None):
    print("Validing")
    img_path='/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/images/COVID-3513.png'
    gt_mask_path='/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/masks/COVID-3513.png'
    img_name=img_path.split("/")[-1].split(".")[0]
    flag=1
    if model is None:
        if flag==1:
            from model import Model
            model=Model().cuda()
            checkpoint=torch.load("./exp/model2_epoch_190.pth")
        else:
            from model2 import Model
            model=Model().cuda()
            checkpoint=torch.load("./exp2/model_epoch_195.pth")
       
        model.load_state_dict(checkpoint['state_dict'])
        # 初始化数据集
    img=Image.open(img_path).convert("RGB")
    gt=Image.open(gt_mask_path).convert("1")
    trans=transforms.Compose([
        transforms.Resize((256,256)),
        TF.to_tensor
    ])
    shutil.copyfile(img_path,
    "./test/{}-img.png".format(img_name)
    )
    shutil.copyfile(gt_mask_path,
    "./test/{}-mask.png".format(img_name)
    )
    
    img=trans(img)
    gt=TF.to_tensor(gt)
    img=img.unsqueeze(0)
    mask=gt.unsqueeze(0)
    print(img.shape)
   

    with torch.no_grad():
        model.eval()
        total_iou=0.0
        total_dice=0.0
        total=0.0
        if flag==1:
            img=img.cuda()
            mask=mask.cuda()
            out = model(img)
            print(out.shape,mask.shape)
            res = out.detach().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res_img=Image.fromarray(res*255)
            res_img.convert('RGB').save("./test/{}-premask.png".format(img_name))
        else:
            img=img.cuda()
            mask=mask.cuda()
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(img)
            res = lateral_map_2.sigmoid().detach().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res_img=Image.fromarray(res*255)
            res_img.convert('RGB').save("./test/{}-premask.png".format(img_name))
        # total_iou+=iou(out,mask)*mask.shape[0]
        # total_dice+=dice_coef(out,mask)*mask.shape[0]
        # total+=mask.shape[0]
        # print("Valid Avg iou: {}".format(total_iou/total))
        # print("Valid Avg dice: {}".format(total_dice/total))  
        return img_name

if __name__ == '__main__':
    create_dataset()
    