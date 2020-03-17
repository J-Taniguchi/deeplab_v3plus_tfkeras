import numpy as np
import tensorflow.keras as keras
from .data_utils import make_xy_from_data_paths
from .data_augment import data_augment
from .label import Label

class path_DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 img_paths,
                 seg_img_paths,
                 image_size,
                 label: Label,
                 batch_size,
                 preprocess=None,
                 augmentation=True,
                 shuffle=True,
                 resize_or_crop="resize",
                 data_type="image"):
        self.img_paths = np.array(img_paths)
        self.seg_img_paths = np.array(seg_img_paths)
        self.image_size = image_size
        self.label = label
        self.batch_size = batch_size
        self.n_images = len(img_paths)
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.shuffle = shuffle
        if resize_or_crop != 'resize' and resize_or_crop != 'crop':
            raise Exception("resize_or_crop must be 'resize' or 'crop'.")
        self.resize_or_crop = resize_or_crop
        self.data_type = data_type

        if (data_type != "image") and (data_type !="index_png") and (data_type !="polygon"):
            raise Exception("data_type must be \"image\" or \"index_png\" or \"polygon\".")

        self.batch_ind = np.arange(self.n_images)

        if shuffle:
            np.random.shuffle(self.batch_ind)

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        tar_ind = self.batch_ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_paths = self.img_paths[tar_ind]
        batch_y_paths = self.seg_img_paths[tar_ind]

        x, y = make_xy_from_data_paths(batch_x_paths,
                                       batch_y_paths,
                                       self.image_size,
                                       self.label,
                                       data_type=self.data_type,
                                       resize_or_crop=self.resize_or_crop)


        if self.augmentation == True:
            x,y = data_augment(x,y,image_size=self.image_size, p=0.95)

        if self.preprocess == None:
            return (x/127.5)-1, y
        else:
            return self.preprocess(x), y

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_ind)

class array_DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 x,
                 y,
                 batch_size,
                 preprocess=None,
                 augmentation=True,
                 shuffle=True,
                 ):
        self.x = np.array(x)
        self.y = np.array(y)
        self.image_size = x.shape[1:3]
        self.batch_size = batch_size
        self.n_images = x.shape[0]
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.shuffle = shuffle

        self.batch_ind = np.arange(self.n_images)

        if shuffle:
            np.random.shuffle(self.batch_ind)

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        tar_ind = self.batch_ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = self.x[tar_ind,:,:,:]
        y = self.y[tar_ind,:,:,:]

        if self.augmentation == True:
            x,y = data_augment(x, y, image_size=self.image_size, p=0.95)

        if self.preprocess == None:
            return (x/127.5)-1, y
        else:
            return self.preprocess(x), y

    def __len__(self):
        return int(np.ceil(self.n_images / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_ind)
