import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json

def inference_large_img(img_path, model, preprocess, label, mode, threshold=0.5, batch_size=8):
    """inference for large image.
        If mode is "simple_crop", simply crop the large image to the model size.
        If mode is "center", only using the center of the prediction.
        If mode is "max_confidence", crop the large image while overrapping, calculate sum of the prediction,
        and, take maximum index as mask.

    Args:
        img_path (path): path for a large image for inference.
        model (Model): trained model
        preprocess (function): preprocessor for the model.
        label (Label):
        mode (str): "simple_crop" or "center" or "max_confidence"
        threshold (float): prediction under this value is treated as "background". Defaults to 0.5.
        batch_size (int): . Defaults to 8.

    Returns:
        np.array, np.array: image array of x and mask.

    """
    image_size = model.input_shape[1:3][::-1]
    image_size_half = (image_size[0]//2, image_size[1]//2)
    image_size_quarter = (image_size[0]//4, image_size[1]//4)

    large_image = Image.open(img_path)
    w_org, h_org = large_image.size

    if mode == "simple_crop":
        nx = np.int(np.ceil(w_org / image_size[0]))
        ny = np.int(np.ceil(h_org / image_size[1]))

        x_inds = np.linspace(0, w_org, nx+1, dtype=int)[0:-1]
        y_inds = np.linspace(0, h_org, ny+1, dtype=int)[0:-1]

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind, y_ind, x_ind+image_size[0], y_ind+image_size[1])
                croped_image = large_image.crop(crop_area)
                croped_image = np.array(croped_image, np.uint8)
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)

        print("predicting")
        y = model.predict(preprocess(x), batch_size=8)

        print("converting y to PIL image")
        y_croped_img_array = np.array(convert_y_to_image_array(y, label, threshold=threshold))

        print("merging image")
        large_mask_image = np.zeros((ny*image_size[1], nx*image_size[0], 3), dtype=x.dtype)
        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]
            now_y = start_index[i,1]
            large_mask_image[now_y:now_y+image_size[1], now_x:now_x+image_size[0],:] = y_croped_img_array[i,:,:,:]
        return np.array(large_image), large_mask_image[0:h_org,0:w_org,:]

    elif mode == "center":
        w = w_org + image_size_half[0]
        h = h_org + image_size_half[1]
        x_inds = np.arange(0, w, image_size_half[0])[0:-1]
        y_inds = np.arange(0, h, image_size_half[1])[0:-1]

        padded_large_image = np.zeros((h,w,3), dtype=np.uint8)
        #put original image to larger black image.
        padded_large_image[image_size_quarter[1]:-image_size_quarter[1],image_size_quarter[0]:-image_size_quarter[0],:] \
            = large_image
        padded_large_image = Image.fromarray(padded_large_image)

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind, y_ind, x_ind+image_size[0], y_ind+image_size[1])
                croped_image = padded_large_image.crop(crop_area)
                croped_image = np.array(croped_image, np.uint8)
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)

        print("predicting")
        y = model.predict(preprocess(x), batch_size=8)

        print("converting y to PIL image")
        croped_y_img_array = np.array(convert_y_to_image_array(y, label, threshold=threshold))

        print("merging image")
        #a little larger size. when crop x, if x is smaller than image_size, paddit with black.
        #so, initial mask must larger than h and w.
        mask_image = np.zeros((h+image_size_quarter[1], w+image_size_quarter[0], 3), dtype=x.dtype)
        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]# + image_size_half[0]
            now_y = start_index[i,1]# + image_size_half[1]
            mask_image[now_y:now_y+image_size_half[1], now_x:now_x+image_size_half[0],:] \
                = croped_y_img_array[i,image_size_quarter[1]:-image_size_quarter[1],image_size_quarter[0]:-image_size_quarter[0],:]

        return np.array(large_image), mask_image[0:h_org,0:w_org,:]
    elif mode == "max_confidence":
        w = w_org + (image_size[0] - 1) * 2
        h = h_org + (image_size[1] - 1) * 2
        x_inds = np.arange(0, w, image_size_quarter[0])[0:-1]
        y_inds = np.arange(0, h, image_size_quarter[1])[0:-1]

        padded_large_image = np.zeros((h,w,3), dtype=np.uint8)
        #put original image to larger black image.
        padded_large_image[(image_size[1]-1):(image_size[1]-1)+h_org,
                           (image_size[0]-1):(image_size[0]-1)+w_org,:] \
            = large_image
        padded_large_image = Image.fromarray(padded_large_image)

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind, y_ind, x_ind+image_size[0], y_ind+image_size[1])
                croped_image = padded_large_image.crop(crop_area)
                croped_image = np.array(croped_image, np.uint8)
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)

        print("predicting")
        y = model.predict(preprocess(x), batch_size=8)
        under_threshold = y[:,:,:,:].max(3) < threshold
        y[under_threshold,:] = 0.0
        y[under_threshold,0] = 1.0

        print("merging image")
        #a little larger size. when crop x, if x is smaller than image_size, paddit with black.
        #so, initial mask must larger than h and w.
        mask = np.zeros((h+image_size[1], w+image_size[0], y.shape[-1]), dtype=np.float32)

        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]# + image_size_half[0]
            now_y = start_index[i,1]# + image_size_half[1]
            mask[now_y:now_y+image_size[1], now_x:now_x+image_size[0],:] \
                += y[i,:,:,:]

        print("converting y to PIL image")
        mask_image = convert_y_to_image_array(mask[np.newaxis,:,:,:], label, threshold=0.0)

        return np.array(large_image), mask_image[0][(image_size[1]-1):(image_size[1]-1)+h_org,
                                                    (image_size[0]-1):(image_size[0]-1)+w_org,:]

    else:
        raise Exception("mode must be \"simple_crop\" or \"center\" or \"max_confidence\"")




'''
def make_x_from_large_img_path(data_path, image_size):
    """crop one large image to image_size.

    Args:
        data_path (str): large image path(must be larger than image_size)
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
'''

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

def convert_y_to_image_array(y, label, threshold=0.5):
    out_img = []
    for i in range(y.shape[0]):
        out_img0 = np.zeros((y.shape[1], y.shape[2], 3), np.uint8)
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
