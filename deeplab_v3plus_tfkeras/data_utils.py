# import copy

import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import h5py
# import numba


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
    image_size_half = (image_size[0] // 2, image_size[1] // 2)
    image_size_quarter = (image_size[0] // 4, image_size[1] // 4)
    n_labels = model.output_shape[-1]

    if type(in_image) is np.ndarray:
        large_image = in_image
    else:
        large_image = cv2.imread(in_image)[:, :, :3]
        large_image = large_image[:, :, ::-1]

    h_org = large_image.shape[0]
    w_org = large_image.shape[1]

    if mode == "simple_crop":
        nx = np.int(np.ceil(w_org / image_size[0]))
        ny = np.int(np.ceil(h_org / image_size[1]))

        x_inds = np.linspace(0, w_org, nx + 1, dtype=int)[0:-1]
        y_inds = np.linspace(0, h_org, ny + 1, dtype=int)[0:-1]

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind,
                             y_ind,
                             x_ind + image_size[0],
                             y_ind + image_size[1])
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
                                :] = croped_image[:, :, :].copy()
                    croped_image = black_image
                x.append(croped_image.copy())

        x = np.array(x)
        start_index = np.array(start_index)

        y = model.predict(preprocess(x), batch_size=batch_size)

        large_y = np.zeros(
            (ny * image_size[1], nx * image_size[0], n_labels),
            dtype=np.float32)
        for i in range(start_index.shape[0]):
            now_x = start_index[i, 0]
            now_y = start_index[i, 1]
            large_y[now_y:now_y + image_size[1],
                    now_x:now_x + image_size[0],
                    :] = y[i, :, :, :]
        return np.array(large_image), large_y[0:h_org, 0:w_org, :]

    elif mode == "center":
        w = w_org + image_size_half[0]
        h = h_org + image_size_half[1]
        x_inds = np.arange(0, w, image_size_half[0])[0:-1]
        y_inds = np.arange(0, h, image_size_half[1])[0:-1]

        padded_large_image = np.zeros((h, w, 3), dtype=np.uint8)
        # put original image to larger black image.
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
                             x_ind + image_size[0],
                             y_ind + image_size[1])
                croped_image = padded_large_image[crop_area[1]:crop_area[3],
                                                  crop_area[0]:crop_area[2], :]
                if (croped_image.shape[0] != image_size[1]) or \
                   (croped_image.shape[1] != image_size[0]):
                    black_image = np.zeros(
                        (image_size[1], image_size[0], 3),
                        dtype=np.uint8)
                    black_image[0:croped_image.shape[0],
                                0:croped_image.shape[1],
                                :] = croped_image[:, :, :].copy()
                    croped_image = black_image
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)
        y = model.predict(preprocess(x), batch_size=batch_size)
        # a little larger size. when crop x, if x is smaller than image_size,
        # paddit with black.
        # so, initial mask must larger than h and w.
        large_y = np.zeros((h + image_size_quarter[1],
                            w + image_size_quarter[0],
                            n_labels), dtype=np.float32)
        for i in range(start_index.shape[0]):
            now_x = start_index[i, 0]  # + image_size_half[0]
            now_y = start_index[i, 1]  # + image_size_half[1]
            large_y[now_y:now_y + image_size_half[1],
                    now_x:now_x + image_size_half[0], :] = \
                y[i,
                  image_size_quarter[1]:-image_size_quarter[1],
                  image_size_quarter[0]:-image_size_quarter[0],
                  :]
        return np.array(large_image), large_y[0:h_org, 0:w_org, :]

    elif mode == "max_confidence":
        w = w_org + (image_size[0] - 1) * 2
        h = h_org + (image_size[1] - 1) * 2
        x_inds = np.arange(0, w, image_size_quarter[0])[0:-1]
        y_inds = np.arange(0, h, image_size_quarter[1])[0:-1]

        padded_large_image = np.zeros((h, w, 3), dtype=np.uint8)
        # put original image to larger black image.
        padded_large_image[(image_size[1] - 1):(image_size[1] - 1) + h_org,
                           (image_size[0] - 1):(image_size[0] - 1) + w_org,
                           :] = large_image

        x = []
        start_index = []
        for i, x_ind in enumerate(x_inds):
            for j, y_ind in enumerate(y_inds):
                start_index.append([x_ind, y_ind])
                crop_area = (x_ind,
                             y_ind,
                             x_ind + image_size[0],
                             y_ind + image_size[1])
                croped_image = padded_large_image[crop_area[1]:crop_area[3],
                                                  crop_area[0]:crop_area[2], :]
                if (croped_image.shape[0] != image_size[1]) or \
                   (croped_image.shape[1] != image_size[0]):
                    black_image = np.zeros(
                        (image_size[1], image_size[0], 3),
                        dtype=np.uint8)
                    black_image[0:croped_image.shape[0],
                                0:croped_image.shape[1],
                                :] = croped_image[:, :, :].copy()
                    croped_image = black_image
                x.append(croped_image)

        x = np.array(x)
        start_index = np.array(start_index)
        y = model.predict(preprocess(x), batch_size=batch_size)
        # a little larger size. when crop x, if x is smaller than image_size, paddit with black.
        # so, initial mask must larger than h and w.
        large_y = np.zeros((h + image_size[1],
                            w + image_size[0],
                            n_labels), dtype=np.float32)
        large_y_count = np.zeros((h + image_size[1],
                                  w + image_size[0],
                                  n_labels), dtype=np.float32)

        for i in range(start_index.shape[0]):
            now_x = start_index[i, 0]  # + image_size_half[0]
            now_y = start_index[i, 1]  # + image_size_half[1]
            large_y[now_y:now_y + image_size[1],
                    now_x:now_x + image_size[0], :] += y[i, :, :, :]
            large_y_count[now_y:now_y + image_size[1],
                          now_x:now_x + image_size[0], :] += 1
        large_y = large_y[(image_size[1] - 1):(image_size[1] - 1) + h_org,
                          (image_size[0] - 1):(image_size[0] - 1) + w_org, :]
        large_y_count = \
            large_y_count[(image_size[1] - 1):(image_size[1] - 1) + h_org,
                          (image_size[0] - 1):(image_size[0] - 1) + w_org, :]
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
                            extra_x_paths=None):
    """make x and y from data paths.

    Args:
        x_paths (list): list of path to x image
        y_paths (list): list of path to y image or json. if None, y is exported as None
        image_size (tuple): model input and output size.(width, height)
        label (Label): class "Label" written in label.py
        data_type (str): select "image" or "index_png" or "polygon"

    Returns:
        np.rray, np.array: x and y

    """
    x = []
    for i, x_path in enumerate(x_paths):
        image = tf.io.read_file(x_path)
        image = tf.image.decode_image(image, channels=3)
        out = np.zeros((image_size[1], image_size[0], 3))
        out[0:image.shape[0], 0:image.shape[1]] = image[:, :]
        x.append(out)
    x = tf.convert_to_tensor(x)
    if y_paths is None:
        return x

    y = []
    for i, y_path in enumerate(y_paths):
        if y_path is None:
            y.append(np.zeros((*image_size[::-1], label.n_labels), np.int32))
            continue
        image = tf.io.read_file(y_path)
        image = tf.image.decode_image(image, channels=3)
        y0 = convert_image_array_to_y(image, label)
        y.append(y0)

    y = tf.convert_to_tensor(y)
    if extra_x_paths is None:
        return x, y
    else:
        extra_x = []
        for i, extra_x_path in enumerate(extra_x_paths):
            if type(extra_x_path) is str:
                out = np.load(extra_x_path)
            else:
                out = np.load(extra_x_path.numpy())
            if len(out.shape) == 2:
                out = out[:, :, np.newaxis]
            extra_x.append(out)
        extra_x = tf.convert_to_tensor(extra_x)

        return x, extra_x, y


def convert_image_array_to_y(image_array, label):
    """Short summary.

    Args:
        image_array (np.array): image y array(batch, y, x, rgb)
        label (Label): class "Label" written in label.py.

    Returns:
        type: .

    """
    y = []
    for i in range(label.n_labels):
        y.append(tf.reduce_all(tf.equal(image_array, label.color[i, :]), axis=2))
    y = tf.stack(y, axis=2)
    return y


def random_crop(image, out_size):
    image_size = image.size
    xmin = np.inf
    ymin = np.inf
    # if fig size is not enough learge, xmin or ymin set to 0.
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
    # if fig size is not enough learge, xmin or ymin set to 0.
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


def save_inference_results(fpath, x, pred, last_activation, y=[], extra_x=[], basenames=[]):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("extra_x", data=extra_x)
        if len(basenames) != 0:
            basenames = [basename.encode("utf8") for basename in basenames]
        f.create_dataset("basenames", data=basenames)

        f.create_dataset("pred", data=pred)
        f.create_dataset("last_activation", data=last_activation)


def load_inference_results(fpath):
    with h5py.File(fpath, "r") as f:
        x = f["x"][()]
        extra_x = f["extra_x"][()]
        y = f["y"][()]
        pred = f["pred"][()]
        basenames = f["basenames"][()]
        last_activation = f["last_activation"][()]
        if len(y) == 0:
            y = None
        if len(basenames) == 0:
            basenames = None
        else:
            basenames = [basename.decode("utf8") for basename in basenames]
        if len(extra_x) == 0:
            extra_x = None

    return x, extra_x, y, pred, basenames, last_activation


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
    df = pd.DataFrame({'label': VOC_CLASSES,
                       'R': VOC_COLORMAP[:, 0],
                       'G': VOC_COLORMAP[:, 1],
                       'B': VOC_COLORMAP[:, 2]})
    df.to_csv('pascal_voc_label.csv', header=False, index=False)
