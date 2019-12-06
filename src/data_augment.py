from albumentations import *
import numpy as np

def augmentor(p, image_size):
    cutout_size = min(image_size) //20
    return Compose([
        #RandomResizedCrop(*image_size, scale=(0.75,1.0), p=0.5),
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        Transpose(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(p=1.0),
            GaussNoise(p=1.0),
        ], p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        #OneOf([
            #OpticalDistortion(p=1.0),
            #GridDistortion(p=1.0),
            #IAAPiecewiseAffine(p=1.0),
            #ElasticTransform(p=1.0),
        #], p=0.5),
        OneOf([
            CLAHE(p=1.0),
            IAASharpen(p=1.0),
            IAAEmboss(p=1.0),
            RandomBrightnessContrast(p=1.0),
            RandomBrightness(p=1.0),
        ], p=0.5),
        OneOf([
            RGBShift(p=1.0),
            ChannelShuffle(p=1.0),
        ], p=0.5),
        #HueSaturationValue(p=0.5),
        #Cutout(max_h_size=cutout_size, max_w_size=cutout_size, p=0.3),
    ], p=p)

def data_augment(images, masks, image_size, p):
    aug = augmentor(p,image_size)
    out_img = []
    out_mask = []
    for i in range(images.shape[0]):
        auged = aug(image=images[i,:,:,:],mask=masks[i,:,:,:])
        out_img.append(auged["image"])
        out_mask.append(auged["mask"])
    return np.array(out_img), np.array(out_mask)
