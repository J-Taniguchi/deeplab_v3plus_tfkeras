import numpy as np
import tensorflow.keras as keras
from image_utils import make_x_from_image_paths, make_y_from_image_paths
from data_augment import data_augment

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 img_paths,
                 seg_img_paths,
                 image_size,
                 label_pd,
                 batch_size,
                 preprocess=None,
                 augmentation=True,
                 shuffle=True,
                 is_index_png=False):
        self.img_paths = np.array(img_paths)
        self.seg_img_paths = np.array(seg_img_paths)
        self.image_size = image_size
        self.label_pd = label_pd
        self.label_np = np.array(label_pd.iloc[:,1:])
        self.batch_size = batch_size
        self.n_images = len(img_paths)
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.shuffle = shuffle
        self.is_index_png = is_index_png

        self.batch_ind = np.arange(self.n_images)

        if shuffle:
            np.random.shuffle(self.batch_ind)

    def __getitem__(self, idx):
        # バッチサイズ分取り出す
        tar_ind = self.batch_ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_paths = self.img_paths[tar_ind]
        batch_y_paths = self.seg_img_paths[tar_ind]

        x = make_x_from_image_paths(batch_x_paths, self.image_size)
        y = make_y_from_image_paths(batch_y_paths, self.image_size, self.label_np, self.is_index_png)

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

'''
def make_data_gen(params, augmentation=True):
    while True :
        batch = make_batch(params)
        for i in range(len(batch)):
            now_img_paths     = [params.img_paths    [batch[i][j]] for j in range(params.batch_size)]
            now_seg_img_paths = [params.seg_img_paths[batch[i][j]] for j in range(params.batch_size)]

            x = make_x_from_image_paths(now_img_paths, params)
            y = make_y_from_image_paths(now_seg_img_paths, params)

            if augmentation == True:
                x,y = data_augment(x,y)

            yield (x/127.5)-1, y

def make_batch(params):
    shuffle_ind = np.random.permutation(np.arange(params.n_images))
    batch = []
    for i in range(params.n_batch):
        s = i * params.batch_size
        e = (i+1) * params.batch_size + 1
        batch.append(shuffle_ind[s:e])
    return batch
'''
