import os
import sys
import h5py
import numpy as np
from glob import glob

def check_data_paths(data_paths, mixed_type_is_error=False):
    is_path = False
    is_h5 = False
    out=[]
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

def make_xy_path_list(data_paths, img_exts=["png"]):
    x = []
    y = []
    for data_path in data_paths:
        x0 = []
        for ext in img_exts:
            x0.extend(glob(os.path.join(data_path, '*.' + ext)))
        x0.sort()
        x.extend(x0)

    for fpath in x:
        base_name = os.path.basename(fpath)
        image_name = os.path.splitext(base_name)[0]
        p = os.path.join(data_path, image_name+'.json')
        if os.path.exists(p):
            y.append(p)
        else:
            y.append(None)
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
