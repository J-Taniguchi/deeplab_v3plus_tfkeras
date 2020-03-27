from albumentations import *
import numpy as np

def augmentor(p, image_size):
    # image_size:(height, width)
    cutout_size = min(image_size) //20
    return Compose([
        #RandomResizedCrop(*image_size, scale=(0.5,2.0), p=0.5),
        #RandomRotate90(p=0.5),
        #Rotate(p=0.5),
        ShiftScaleRotate(rotate_limit=20,
                         scale_limit=0.1,
                         shift_limit=0.1,
                         border_mode=0,
                         p=1.0),
        #OneOf([
            #ShiftScaleRotate(rotate_limit=45, scale_limit=0.2, shift_limit=0.1, interpolation=0, p=1.0),
            #ShiftScaleRotate(rotate_limit=45, scale_limit=0.2, shift_limit=0.1, interpolation=1, p=1.0),
            #], p=0.8),
        Flip(p=0.5),
        #Transpose(p=0.5),
        #Downscale(scale_min=0.25, scale_max=0.8, p=0.3),
        OneOf([
            IAAAdditiveGaussianNoise(p=1.0),
            GaussNoise(p=1.0),
            JpegCompression(p=1.0,quality_lower=80, quality_upper=100),
            Downscale(p=1.0, scale_min=0.9, scale_max=0.99)
            ], p=0.5),
        #OneOf([
            #OpticalDistortion(p=1.0),
            #GridDistortion(p=1.0),
            #IAAPiecewiseAffine(p=1.0),
            #ElasticTransform(p=1.0),
        #], p=0.5),
        OneOf([
            #CLAHE(p=1.0),
            #IAASharpen(p=1.0),
            #IAAEmboss(p=1.0),
            RandomBrightnessContrast(p=1.0),
            RandomGamma(p=1.0),
            RandomContrast(p=1.0),
            RandomBrightness(p=1.0),
        ], p=0.5),
        #OneOf([
            #RGBShift(p=1.0),
            #ChannelShuffle(p=1.0),
        #], p=0.5),
        RGBShift(p=0.5),
        HueSaturationValue(p=0.5),
        #RandomShadow(p=0.5),
        #Cutout(max_h_size=cutout_size, max_w_size=cutout_size, p=0.3),
    ], p=p)

def data_augment(images, masks, image_size, p):
    # image_size:(height, width)
    aug = augmentor(p, image_size)
    out_img = []
    out_mask = []
    #if len(images.shape) != 4:
        #raise Exception("dimension of images for data_augment must be 4")
    #if len(masks.shape) != 4:
        #raise Exception("dimension of masks for data_augment must be 4")
    for i in range(images.shape[0]):
        auged = aug(image=images[i,:,:,:], mask=masks[i,:,:,:])
        out_img.append(auged["image"])
        out_mask.append(auged["mask"])
    return np.array(out_img), np.array(out_mask)
