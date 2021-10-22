import cv2
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision import models, transforms
from torchcam import utils, cams

def cal_iou(img, map, label):
    """calculate intersection over union

    Parameters
    ----------
    img : PIL.PngImagePlugin.PngImageFile
        pillow image
    map : numpy.ndarray
        feature importance map
    label : numpy.ndarray
        sementic segmentation label
    """
    result = utils.overlay_mask(img, 
                        transforms.functional.to_pil_image(map, mode='F'), 
                        alpha=0.5
                        )

    trans = transforms.ToPILImage()
    map = trans(map)
    map = np.array(map.resize((img.width, img.height)))
    pred = np.where(map > 255*0.6, True, False)

    iou = np.sum(np.logical_and(pred, label))/np.sum(np.logical_or(pred, label))

    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("IOU: ", iou)

def cal_auc(img, map, class_number:int, reverse=False):
    """calculate area under curve

    Parameters
    ----------
    img : PIL.PngImagePlugin.PngImageFile
        pillow image
    map : numpy.ndarray
        feature importance map
    class_number: int
        ground truth label
    reverse : bool, optional
        set importance order ascending or descending, by default False(descending)
    """
    input = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        out = model(input)
    precision = torch.sigmoid(out)
    
    precisions = list()
    precisions.append( precision[0, class_indx])
    map = np.array(map)
    map = cv2.resize(map, 
                    (img.width, img.height), 
                    fx=0, 
                    fy=0, 
                    interpolation = cv2.INTER_NEAREST
                    )
    if not reverse:
        mask_list = np.unique(map)[::-1]
    else:
        mask_list = np.unique(map)
    mean_img = int(np.mean(img))

    for mask in mask_list:
        img = np.array(img)
        img = np.where((np.repeat(map[:, :, np.newaxis], 3, axis=2) == mask), 
                    mean_img, 
                    img
                    )
        input = transforms.ToTensor()(img).unsqueeze(0)
        
        with torch.no_grad():
            out = model(input)
        
        precisions.append(torch.sigmoid(out)[0, class_indx])
    
    plt.plot(precisions)
    plt.show()

    print("AUC: ", np.sum(precisions)*(1/len(precisions)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('xai_basic', add_help=False)
    parser.add_argument('--image_dir', default='./image/00001.png', type=str)
    parser.add_argument('--label_dir', default='./label/00001.png', type=str)
    parser.add_argument('--model_dir', default='./model.ckpt', type=str)
    parser.add_argument('--target_class', default=0, type=int, help='0: Black Sea Sprat,\
                                                                     1: Gilt-Head Bream\
                                                                     2: Hourse Mackerel\
                                                                     3: Red Mullet\
                                                                     4: Red Sea Bream\
                                                                     5: Sea Bass\
                                                                     6: Shrimp\
                                                                     7: Striped Red Mullet\
                                                                     8: Trout')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model.load_state_dict(torch.load(args.model_dir, map_location=device))
    model.eval()

    cam = cams.CAM(model, 'layer4', 'fc')

    img = Image.open(args.image_dir)
    input = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        out = model(input)

    class_indx = args.target_class
    map = cam(class_idx = class_indx)
    label = np.array(Image.open(args.label_dir))

    cal_iou(img, map, label)
    cal_auc(img.copy(), map, class_indx, True)