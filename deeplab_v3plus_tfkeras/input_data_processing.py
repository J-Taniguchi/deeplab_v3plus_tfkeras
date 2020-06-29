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

# TODO: x_pathsというよりx_dirs
def make_data_path_list(
    x_paths,
    y_paths=None,
    extra_x_paths=None,
    img_exts=["png", "jpg"],
    return_basenames=False):
    x = []
    y = []
    if return_basenames:
        basenames = []

    if extra_x_paths is not None:
        extra_x = []
    for i, x_path in enumerate(x_paths):
        if y_paths is not None:
            y_path = y_paths[i]
        if extra_x_paths is not None:
            extra_x_path = extra_x_paths[i]

        x0 = []
        for ext in img_exts:
            x0.extend(glob(os.path.join(x_path, '*.' + ext)))
        x0.sort()

        for fpath in x0:
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            if y_paths is not None:
                p = os.path.join(y_path, base_name + '.png')
                if os.path.exists(p):
                    y.append(p)
                else:
                    raise Exception("{} dose not exist.".format(p))
            if extra_x_paths is not None:
                p = os.path.join(extra_x_path, base_name + '.npy')
                if os.path.exists(p):
                    extra_x.append(p)
                else:
                    raise Exception("{} dose not exist.".format(p))
            if return_basenames:
                basenames.append(base_name)
        x.extend(x0)

    returns = [x]
    if y_paths is not None:
        returns.append(y)
    if extra_x_paths is not None:
        returns.append(extra_x)
    if return_basenames:
        returns.append(basenames)
    # return order is (x, y, extra_x, basenames)
    return tuple(returns)


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
