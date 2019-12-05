import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json

def make_x_from_data_paths(data_paths, image_size):
    x = []
    for i, data_path in enumerate(data_paths):
        image = read_image(data_path, image_size)
        image = np.array(image, np.uint8)
        x.append(image)
    return np.array(x)

def make_y_from_data_paths(data_paths, image_size, label, data_type="image"):
    y = []
    for i, data_path in enumerate(data_paths):
        if data_type == "image":
            image = read_image(data_path, image_size)
            image = np.array(image, np.int32)[:,:,:3]
            y0 = convert_image_array_to_y(image, label)
            y.append(y0)
        elif data_type == "index_png":
            image = read_image(data_path, image_size)
            image = image.convert('RGB')
            image = np.array(image, np.int32)[:,:,:3]
            y0 = convert_image_array_to_y(image, label)
            y.append(y0)
        elif data_type == "polygon":
            y0 = make_y_from_poly_json_path(data_path, image_size, label)
            y.append(y0)
        else:
            raise Exception("data_type must be \"image\" or \"index_png\" or \"polygon\".")
    return np.array(y)

def make_y_from_poly_json_path(data_path, image_size, label):
    y = np.empty((*image_size, label.n_labels), np.float32)

    if data_path == None:
        for i in range(label.n_labels):
            if i == 0:
                y[:,:,i] = np.ones(image_size, np.float32)
            else:
                y[:,:,i] = np.zeros(image_size, np.float32)
    else:
        with open(data_path) as d:
            poly_json = json.load(d)
        n_poly = len(poly_json['shapes'])

        images = []
        draws  = []
        for i in range(label.n_labels):
            if i == 0: #背景は全部Trueにしておいて，何らかのオブジェクトがある場合にFalseでぬる
                images.append(Image.new(mode='1', size=image_size, color=True))
            else:
                images.append(Image.new(mode='1', size=image_size, color=False))
            draws.append(ImageDraw.Draw(images[i]))

        for i in range(n_poly):
            label_name = poly_json['shapes'][i]['label']
            label_num = label.name.index(label_name)
            if label_num == 0: #背景のポリゴンは作成していないはずなので，何も通らないはず．
                pass
            else:
                poly = poly_json['shapes'][i]['points']
                poly = tuple(map(tuple, poly))
                draws[label_num].polygon(poly, fill=True)
                #背景は全部Trueにしておいて，何らかのオブジェクトがある場合にFalseでぬる
                draws[0].polygon(poly, fill=False)

        for i in range(label.n_labels):
            y[:,:,i] = np.array(images[i])
    return y

def convert_image_array_to_y(image_array, label):
    y = np.zeros((*(image_array).shape[:2], label.n_labels), np.float32)
    for i in range(label.n_labels):
        y[:,:,i] = np.all(np.equal(image_array , label.color[i,:]), axis=2).astype(np.float32)
    return y

def read_image(img_path, out_img_size):
    image = Image.open(img_path)
    image = image.resize(out_img_size)
    return image

#要動作確認
def convert_y_to_image_array(y, image_size, label, threshold=0.0):
    out_img = []
    for i in range(y.shape[0]):
        out_img0 = np.zeros((*image_size, 3), np.float32)
        under_threshold = y[i,:,:,:].max(2) < threshold
        y[i,under_threshold,0] = 0.0
        max_category = y[i,:,:,:].argmax(2)
        for j in range(label.n_labels):
            out_img0[max_category==j] = label.color[j,:]/255
        out_img.append(out_img0)
    return out_img

def make_pascal_voc_label_csv():
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
    VOC_COLORMAP = np.array(VOC_COLORMAP)
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    df = pd.DataFrame({ 'label' : VOC_CLASSES,
                        'R' : VOC_COLORMAP[:,0],
                        'G' : VOC_COLORMAP[:,1],
                        'B' : VOC_COLORMAP[:,2]
                         })
    df.to_csv('pascal_voc_label.csv', header=False, index=False)
