import torch
from torch.autograd import Function
import numpy as np
def iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  
    b = pred.shape[0]
    iou = 0.0

    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
        IoU1 = Iand1/Ior1

        # IoU loss is (1-IoU1)
        # IoU = IoU + (1 - IoU1)
        iou = iou + IoU1

    return iou/b


class IOU(torch.nn.Module):
  
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return iou(pred, target)

class DiceCoeff(Function):


    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(input_, target)
        eps = 0.0001
        self.inter = torch.dot(input_.view(-1), target.view(-1))
        self.union = torch.sum(input_) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output: torch.Tensor) -> tuple:

        input_, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
   
    if inputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(inputs, targets)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def dice_coeff_metric(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()

    if target.sum() == 0 and inputs.sum() == 0:
        return torch.tensor(1.)

    return intersection / union


def dice_coef(preds, trues, smooth=1e-3):
    preds = preds.contiguous().view(preds.size(0), -1).float()
    trues = trues.contiguous().view(preds.size(0), -1).float()
    inter = torch.sum(preds * trues, dim=1)
    dice = torch.mean((2.0 * inter + smooth) / (preds.sum(dim=1) + trues.sum(dim=1) + smooth))
    return dice


def dice_coef_numpy(preds, trues, smooth=1e-3, channel=None):
    if channel is not None:
        preds = preds[:, channel, :, :]
        trues = trues[:, channel, :, :]
    preds = preds.reshape(preds.shape[0], -1)
    trues = trues.reshape(trues.shape[0], -1)

    inter = np.sum(preds * trues, 1)
    dice = np.mean((2.0 * inter + smooth) / (preds.sum(1) + trues.sum(1) + smooth))
    return dice

def test():
    # a=torch.randn((8,1,256,256))
    # b=torch.randn((8,1,256,256))
    # iou_scroe=iou(a,b)
    # dice_score=dice_coef(a,b)
    # print(iou_scroe,dice_score)
    from PIL import Image
    import torchvision.transforms.functional as TF
    img=Image.open("/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/masks/COVID-1.png").convert('1')
    img2=Image.open("/home/user1/homework/yxyxfx/COVID-19_Radiography_Dataset/COVID/masks/COVID-2.png").convert('1')
    trans=TF.to_tensor
    img=trans(img)
    img2=trans(img2)
    img=img.unsqueeze(0)
    img2=img2.unsqueeze(0)
    img11=torch.cat([img,img2,img,img2],dim=0)
    img22=torch.cat([img2,img2,img,img2],dim=0)
    # print(img.shape,img.shape,img[0].shape)


    score=dice_coef(img22,img11)
    iouii=iou(img11,img22)
    print(score,iouii)

if __name__ == '__main__':
    test()
    