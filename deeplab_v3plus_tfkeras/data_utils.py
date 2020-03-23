import numpy as np
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import json
import h5py
import copy
import numba

def inference_large_img(in_image,
                        model,
                        preprocess,
                        mode,
                        threshold=0.5,
                        batch_size=8):
    """inference for large image.
        If mode is "simple_crop",    simply crop the large image
                                     to the model size.
        If mode is "center",         only using the center of the prediction.
        If mode is "max_confidence", crop the large image while overrapping,
                                     calculate sum of the prediction,
                                     and take maximum index as mask.

    Args:
        in_image (path or np.array): path for a large image for inference.
        model (Model):               trained model
        preprocess (function):       preprocessor for the model.
        mode (str):                 "simple_crop" or "center" or
                                    "max_confidence"
        threshold (float):          prediction under this value is treated as
                                    "background". Defaults to 0.5.
        batch_size (int):           Defaults to 8.

    Returns:
        np.array, np.array: image array of input_image and pred.

    """
    image_size = model.input_shape[1:3][::-1]
    image_size_half = (image_size[0]//2, image_size[1]//2)
    image_size_quarter = (image_size[0]//4, image_size[1]//4)
    n_labels = model.output_shape[-1]

    if type(in_image) is np.ndarray:
        large_image = in_image
    else:
        large_image = cv2.imread(in_image)[:,:,:3]
        large_image = large_image[:,:,::-1]

    h_org = large_image.shape[0]
    w_org = large_image.shape[1]

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
                crop_area = (x_ind,
                             y_ind,
                             x_ind+image_size[0],
                             y_ind+image_size[1])
                croped_image = large_image[crop_area[1]:crop_area[3],
                                           crop_area[0]:crop_area[2],
                                           :]
                if (croped_image.shape[0] != image_size[1]) or \
                    (croped_image.shape[1] != image_size[0]):
                    black_image = np.zeros(
                        (image_size[1], image_size[0], 3),
                        dtype=np.uint8)
                    black_image[0:croped_image.shape[0],
                                0:croped_image.shape[1],
                                :] = croped_image[:,:,:].copy()
                    croped_image = black_image
                x.append(croped_image.copy())

        x = np.array(x)
        start_index = np.array(start_index)

        y = model.predict(preprocess(x), batch_size=batch_size)

        large_y = np.zeros(
            (ny*image_size[1], nx*image_size[0], n_labels),
            dtype=np.float32)
        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]
            now_y = start_index[i,1]
            large_y[now_y:now_y+image_size[1],
                    now_x:now_x+image_size[0],
                    :] = y[i,:,:,:]
        return np.array(large_image), large_y[0:h_org,0:w_org,:]




    elif mode == "center":
        w = w_org + image_size_half[0]
        h = h_org + image_size_half[1]
        x_inds = np.arange(0, w, image_size_half[0])[0:-1]
        y_inds = np.arange(0, h, image_size_half[1])[0:-1]

        padded_large_image = np.zeros((h,w,3), dtype=np.uint8)
        #put original image to larger black image.
        padded_large_image[image_size_quarter[1]:-image_size_quarter[1],
                           image_size_quarter[0]:-image_size_quarter[0],
                           :] = large_image

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind,
                             y_ind,
                             x_ind+image_size[0],
                             y_ind+image_size[1])
                croped_image = padded_large_image[crop_area[1]:crop_area[3],
                                                  crop_area[0]:crop_area[2],:]
                if (croped_image.shape[0] != image_size[1]) or \
                    (croped_image.shape[1] != image_size[0]):
                    black_image = np.zeros(
                        (image_size[1], image_size[0], 3),
                        dtype=np.uint8)
                    black_image[0:croped_image.shape[0],
                                0:croped_image.shape[1],
                                :] = croped_image[:,:,:].copy()
                    croped_image = black_image
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)
        y = model.predict(preprocess(x), batch_size=batch_size)
        # a little larger size. when crop x, if x is smaller than image_size,
        # paddit with black.
        # so, initial mask must larger than h and w.
        large_y = np.zeros((h+image_size_quarter[1],
                            w+image_size_quarter[0],
                            n_labels), dtype=np.float32)
        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]# + image_size_half[0]
            now_y = start_index[i,1]# + image_size_half[1]
            large_y[now_y:now_y+image_size_half[1],
                    now_x:now_x+image_size_half[0], :] = \
                y[i,
                  image_size_quarter[1]:-image_size_quarter[1],
                  image_size_quarter[0]:-image_size_quarter[0],
                  :]
        return np.array(large_image), large_y[0:h_org,0:w_org,:]

    elif mode == "max_confidence":
        w = w_org + (image_size[0] - 1) * 2
        h = h_org + (image_size[1] - 1) * 2
        x_inds = np.arange(0, w, image_size_quarter[0])[0:-1]
        y_inds = np.arange(0, h, image_size_quarter[1])[0:-1]

        padded_large_image = np.zeros((h,w,3), dtype=np.uint8)
        #put original image to larger black image.
        padded_large_image[(image_size[1]-1):(image_size[1]-1)+h_org,
                           (image_size[0]-1):(image_size[0]-1)+w_org,
                           :] = large_image

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind,
                             y_ind,
                             x_ind+image_size[0],
                             y_ind+image_size[1])
                croped_image = padded_large_image[crop_area[1]:crop_area[3],
                                                  crop_area[0]:crop_area[2],:]
                if (croped_image.shape[0] != image_size[1]) or \
                    (croped_image.shape[1] != image_size[0]):
                    black_image = np.zeros(
                        (image_size[1], image_size[0], 3),
                        dtype=np.uint8)
                    black_image[0:croped_image.shape[0],
                                0:croped_image.shape[1],
                                :] = croped_image[:,:,:].copy()
                    croped_image = black_image
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)
        y = model.predict(preprocess(x), batch_size=batch_size)
        #a little larger size. when crop x, if x is smaller than image_size, paddit with black.
        #so, initial mask must larger than h and w.
        large_y = np.zeros((h+image_size[1],
                            w+image_size[0],
                            n_labels), dtype=np.float32)
        large_y_count = np.zeros((h+image_size[1],
                                  w+image_size[0],
                                  n_labels), dtype=np.float32)

        for i in range(start_index.shape[0]):
            now_x = start_index[i,0]# + image_size_half[0]
            now_y = start_index[i,1]# + image_size_half[1]
            large_y[now_y:now_y+image_size[1],
                    now_x:now_x+image_size[0],:] += y[i,:,:,:]
            large_y_count[now_y:now_y+image_size[1],
                          now_x:now_x+image_size[0],:] += 1
        large_y = large_y[(image_size[1]-1):(image_size[1]-1)+h_org,
                          (image_size[0]-1):(image_size[0]-1)+w_org, :]
        large_y_count = \
            large_y_count[(image_size[1]-1):(image_size[1]-1)+h_org,
                          (image_size[0]-1):(image_size[0]-1)+w_org, :]
        if large_y_count.min() <= 0:
            raise Exception("SOMTING WRONG")
        large_y = large_y / large_y_count
        return np.array(large_image), large_y
    elif mode == "whole":
        x = large_image
        y = model.predict(preprocess(x), batch_size=batch_size)
        return x, y

    else:
        raise Exception("mode must be \"simple_crop\" or \"center\" or \"max_confidence\"")


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
        elif resize_or_crop is False:
            pass
        else:
            raise Exception("resize_or_crop must be 'resize' or 'crop'.")
        image = np.array(image, np.uint8)[:,:,0:3]
        x.append(image)
    x = np.array(x)

    if y_paths is None:
        return x, None

    y = []
    for i, y_path in enumerate(y_paths):
        if y_path is None:
            y.append(np.zeros((*image_size[::-1], label.n_labels),np.int32))
            continue

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
    y = np.empty((image_size[1], image_size[0], label.n_labels), np.float32)

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
            if (i == 0) and (label.name[0]=="background") : #背景は全部Trueにしておいて，何らかのオブジェクトがある場合にFalseでぬる
                images.append(Image.new(mode='1', size=org_image_size, color=True))
            else:
                images.append(Image.new(mode='1', size=org_image_size, color=False))
            draws.append(ImageDraw.Draw(images[i]))

        for i in range(n_poly):
            label_name = poly_json['shapes'][i]['label']
            label_num = label.name.index(label_name)

            poly = poly_json['shapes'][i]['points']
            poly = tuple(map(tuple, poly))
            draws[label_num].polygon(poly, fill=True)
            #背景は全部Trueにしておいて，何らかのオブジェクトがある場合にFalseでぬる
            if label.name[0] == "background":
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

def save_inference_results(fpath, x, pred, last_activation, y=None):
    with h5py.File(fpath,"w") as f:
        if y is None:
            f.create_dataset("have_y", data=False)
        else:
            f.create_dataset("have_y", data=True)
        # this case means inference for different image sizes.
        if x.dtype == 'O':
            f.create_dataset("dataset_type", data="list")
            f.create_dataset("n_images", data=len(x))
            for i in range(len(x)):
                f.create_dataset("x/{}".format(i), data=x[i])
                f.create_dataset("pred/{}".format(i), data=pred[i])

                if y is not None:
                    f.create_dataset("y/{}".format(i), data=y[i])
                else:
                    if i == 0:
                        f.create_dataset("y", data=y)

        else:
            f.create_dataset("dataset_type", data="array")
            f.create_dataset("x", data=x)
            f.create_dataset("y", data=y)
            f.create_dataset("pred", data=pred)
        f.create_dataset("last_activation", data=last_activation)

def load_inference_results(fpath):
    with h5py.File(fpath, "r") as f:
        dataset_type = f["dataset_type"][()]
        have_y = f["have_y"][()]
        print(dataset_type)
        if dataset_type == "list":
            x = []
            if have_y:
                y = []
            pred = []
            n_images = f["n_images"][()]
            for i in range(n_images):
                x.append(f["x/{}".format(i)][()])
                pred.append(f["pred/{}".format(i)][()])
                if have_y:
                    y.append(f["y/{}".format(i)][()])
                else:
                    if i == 0:
                        y = f["y"][()]

        else:
            x = f["x"][()]
            y = f["y"][()]
            pred = f["pred"][()]
        last_activation = f["last_activation"][()]
    return x, y, pred, last_activation

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
