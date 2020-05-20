import os
# import sys

import h5py
import numpy as np
from glob import glob


def check_data_paths(data_paths, mixed_type_is_error=False):
    is_path = False
    is_h5 = False
    out = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            is_path = True
            out.append("dir")
        elif os.path.isfile(data_path):
            is_h5 = True
            out.append("h5")

    if mixed_type_is_error and is_path and is_h5:
        print(data_paths)
        raise Exception("all data_paths must same type (dir or h5).")

    return out


def make_xy_path_list(x_paths, y_paths, img_exts=["png", "jpg"]):
    x = []
    y = []
    for i, x_path in enumerate(x_paths):
        if y_paths is not None:
            y_path = y_paths[i]

        x0 = []
        for ext in img_exts:
            x0.extend(glob(os.path.join(x_path, '*.' + ext)))
        x0.sort()

        if y_paths is not None:
            for fpath in x0:
                base_name = os.path.basename(fpath)
                image_name = os.path.splitext(base_name)[0]
                p = os.path.join(y_path, image_name + '.png')
                if os.path.exists(p):
                    y.append(p)
                else:
                    x0.remove(fpath)
                    print("{} dose not exist. {} is removed from dataset.".format(p, fpath))
        x.extend(x0)
    return x, y


def make_xy_array(data_paths):
    for i, data_path in enumerate(data_paths):
        if i == 0:
            with h5py.File(data_path, "r") as f:
                x = f["x"][:].astype(np.uint8)
                y = f["y"][:].astype(np.float32)
        else:
            with h5py.File(data_path, "r") as f:
                x0 = f["x"][:].astype(np.uint8)
                y0 = f["y"][:].astype(np.float32)
            x = np.concatenate([x, x0], axis=0)
            y = np.concatenate([y, y0], axis=0)
    return x, y
