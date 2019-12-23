import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json


def make_x_from_large_img_path(data_path, image_size):
    """crop one large image to image_size.

    Args:
        data_path (str): large image path(larger than image_size)
        image_size (tuple): input image size.

    Returns:
        x(np.array): (batch,y,x,rgb)
        start_index(np.array): (batch_ind,x_start, y_start) indicates the location in the image.
                               batch_ind is corresponding to 1st axis of x.

    """
    x = []
    start_index = []
    large_image = Image.open(data_path)
    w,h = large_image.size
    nx = np.int(np.ceil(w / image_size[0]))
    ny = np.int(np.ceil(h / image_size[1]))

    x_inds = np.linspace(0, w, nx+1, dtype=int)[0:-1]
    y_inds = np.linspace(0, h, ny+1, dtype=int)[0:-1]

    for i, x_ind in enumerate(x_inds):
        for j, y_ind in enumerate(y_inds):
            start_index.append([x_ind, y_ind])
            crop_area = (x_ind, y_ind, x_ind+image_size[0], y_ind+image_size[1])
            croped_image = large_image.crop(crop_area)
            croped_image = np.array(croped_image, np.uint8)
            x.append(croped_image)

    return np.array(x), np.array(start_index)

#overlapを考えたものに書き換えたい
def merge_croped_large_image(x, image_size, start_ind):
    nx = np.unique(start_ind[:,0]).size
    ny = np.unique(start_ind[:,1]).size
    #large_image = np.zeros((ny*image_size[1], nx*image_size[0], 3), dtype=np.uint8)
    large_image = np.zeros((ny*image_size[1], nx*image_size[0], 3), dtype=x.dtype)

    for i in range(start_ind.shape[0]):
        now_x = start_ind[i,0]
        now_y = start_ind[i,1]
        large_image[now_y:now_y+image_size[1], now_x:now_x+image_size[0],:] = x[i,:,:,:]
    return large_image

def make_xy_from_data_paths(x_paths,
                            y_paths,
                            image_size,
                            label,
                            data_type,
                            resize_or_crop="resize"):
    """make x and y from data paths.

    Args:
        x_paths (list): list of path to x image
        y_paths (list): list of path to y image or json. if None, y is exported as None
        image_size (tuple): model input and output size.(width, height)
        label (Label): class "Label" written in label.py
        data_type (str): select "image" or "index_png" or "polygon"
        resize_or_crop (str): select "resize" or "crop". Defaults to "resize".

    Returns:
        np.rray, np.array: x and y

    """
    x = []
    crop_areas = []
    for i, x_path in enumerate(x_paths):
        image = Image.open(x_path)
        if resize_or_crop == "resize":
            image = image.resize(image_size)
        elif resize_or_crop == "crop":
            crop_area = get_random_crop_area(image.size, image_size)
            image = image.crop(crop_area)
            crop_areas.append(crop_area)
        else:
            raise Exception("resize_or_crop must be 'resize' or 'crop'.")
        image = np.array(image, np.uint8)
        x.append(image)
    x = np.array(x)

    if y_paths is None:
        return x, None

    y = []
    for i, y_path in enumerate(y_paths):
        if resize_or_crop == "resize":
            if data_type == "image":
                image = Image.open(y_path)
                image = image.resize(image_size)
                image = np.array(image, np.int32)[:,:,:3]
                y0 = convert_image_array_to_y(image, label)
                y.append(y0)
            elif data_type == "index_png":
                image = Image.open(y_path)
                image = image.resize(image_size)
                image = image.convert('RGB')
                image = np.array(image, np.int32)[:,:,:3]
                y0 = convert_image_array_to_y(image, label)
                y.append(y0)
            elif data_type == "polygon":
                y0 = make_y_from_poly_json_path(y_path, image_size, label)
                y.append(y0)
            else:
                raise Exception("data_type must be \"image\" or \"index_png\" or \"polygon\".")
        elif resize_or_crop == "crop":
            if data_type == "image":
                image = Image.open(y_path)
                image = image.crop(crop_areas[i])
                image = np.array(image, np.int32)[:,:,:3]
                y0 = convert_image_array_to_y(image, label)
                y.append(y0)
            elif data_type == "index_png":
                image = Image.open(y_path)
                image = image.crop(crop_areas[i])
                image = image.convert('RGB')
                image = np.array(image, np.int32)[:,:,:3]
                y0 = convert_image_array_to_y(image, label)
                y.append(y0)
            elif data_type == "polygon":
                y0 = make_y_from_poly_json_path(y_path, image_size, label, crop_areas[i])
                y.append(y0)
            else:
                raise Exception("data_type must be \"image\" or \"index_png\" or \"polygon\".")

    y = np.array(y)
    return x, y

def make_y_from_poly_json_path(data_path,
                               image_size,
                               label,
                               crop_area=None):
    """Short summary.

    Args:
        data_path (path): path to json file.
        image_size (tuple): model input and output size.(width, height)
        label (Label): class "Label" written in label.py
        crop_area (None or tuple): If None, resize.
                                   If tuple(xmin, ymin, xmax, ymax), crop to this area.
                                   Defaults to None.

    Returns:
        np.array: y

    """
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
        org_image_size = (poly_json["imageWidth"], poly_json["imageHeight"])
        n_poly = len(poly_json['shapes'])

        images = []
        draws  = []
        for i in range(label.n_labels):
            if i == 0: #背景は全部Trueにしておいて，何らかのオブジェクトがある場合にFalseでぬる
                images.append(Image.new(mode='1', size=org_image_size, color=True))
            else:
                images.append(Image.new(mode='1', size=org_image_size, color=False))
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
            if crop_area is None:
                y[:,:,i] = np.array(images[i].resize(image_size))
            else:
                y[:,:,i] = np.array(images[i].crop(crop_area))
    return y

def convert_image_array_to_y(image_array, label):
    """Short summary.

    Args:
        image_array (np.array): image y array(batch, y, x, rgb)
        label (Label): class "Label" written in label.py.

    Returns:
        type: .

    """
    y = np.zeros((*(image_array).shape[:2], label.n_labels), np.float32)
    for i in range(label.n_labels):
        y[:,:,i] = np.all(np.equal(image_array , label.color[i,:]), axis=2).astype(np.float32)
    return y

def random_crop(image, out_size):
    image_size = image.size
    xmin = np.inf
    ymin = np.inf
    #if fig size is not enough learge, xmin or ymin set to 0.
    if out_size[0] >= image_size[0]:
        xmin = 0
    if out_size[1] >= image_size[1]:
        ymin = 0

    if xmin == np.inf:
        x_res = image_size[0] - out_size[0]
        xmin = np.random.choice(np.arange(0, x_res + 1))
    if ymin == np.inf:
        y_res = image_size[1] - out_size[1]
        ymin = np.random.choice(np.arange(0, y_res + 1))
    xmax = xmin + out_size[0]
    ymax = ymin + out_size[1]
    return image.crop((xmin, ymin, xmax, ymax))

def get_random_crop_area(image_size, out_size):
    xmin = np.inf
    ymin = np.inf
    #if fig size is not enough learge, xmin or ymin set to 0.
    if out_size[0] >= image_size[0]:
        xmin = 0
    if out_size[1] >= image_size[1]:
        ymin = 0

    if xmin == np.inf:
        x_res = image_size[0] - out_size[0]
        xmin = np.random.choice(np.arange(0, x_res + 1))
    if ymin == np.inf:
        y_res = image_size[1] - out_size[1]
        ymin = np.random.choice(np.arange(0, y_res + 1))
    xmax = xmin + out_size[0]
    ymax = ymin + out_size[1]
    return (xmin, ymin, xmax, ymax)

def convert_y_to_image_array(y, image_size, label, threshold=0.5):
    out_img = []
    for i in range(y.shape[0]):
        out_img0 = np.zeros((*image_size, 3), np.uint8)
        under_threshold = y[i,:,:,:].max(2) < threshold
        y[i,under_threshold,0] = 1.0
        max_category = y[i,:,:,:].argmax(2)
        for j in range(label.n_labels):
            out_img0[max_category==j] = label.color[j,:]
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
