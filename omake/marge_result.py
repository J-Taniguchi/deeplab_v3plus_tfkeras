import os
from glob import glob

from tqdm import tqdm

import numpy as np
import cv2

img_size = 256
tar_root_dir = "../result/case1"
out_root_dir = "../result/case1/marged"


def get_xy_from_path(p, suffix):
    basename = ".".join(os.path.basename(p).split(".")[0:-1])
    ll = len(suffix) + 1
    basename = basename[0:-ll]
    x, y = basename.split("_")[-2:]
    return int(x), int(y)


def marge_images(target_dir, target_imgs, out_dir):
    x_imgs = glob(os.path.join(target_dir, "*_x.png"))

    prefixes = []
    for img in x_imgs:
        basename = ".".join(os.path.basename(img).split(".")[0:-1])
        p = "_".join(basename.split("_")[0:-3])
        if p not in prefixes:
            prefixes.append(p)

    for prefix in prefixes:
        # get target x img paths
        tar_x_imgs = []
        for img in x_imgs:
            if os.path.basename(img).startswith(prefix):
                tar_x_imgs.append(img)
        # get out fig size
        x_max = 0
        y_max = 0
        for img in tar_x_imgs:
            x, y = get_xy_from_path(img, suffix="x")

            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
        x_max += img_size
        y_max += img_size
        for target_img in tqdm(target_imgs):
            out_fpath = os.path.join(out_dir, "{}_{}.png".format(prefix, target_img))
            out_fig = np.ones((y_max, x_max, 3), dtype=np.uint8) * 255
            tar_imgs = glob(os.path.join(target_dir, "{}_*_{}.png".format(prefix, target_img)))
            for img in tar_imgs:
                x, y = get_xy_from_path(img, suffix=target_img)
                out_fig[y: y + img_size, x: x + img_size, :] = cv2.imread(img)

            cv2.imwrite(out_fpath, out_fig)


# train
target_dir = os.path.join(tar_root_dir, "train")
out_dir = os.path.join(out_root_dir, "train")
os.makedirs(out_dir, exist_ok=True)
target_imgs = [
    "x",
    "pred_seg",
    "pred_x_seg",
    "true_seg",
    "true_x_seg"
]

marge_images(target_dir, target_imgs, out_dir)


# test
target_dir = os.path.join(tar_root_dir, "test_test_x")
out_dir = os.path.join(out_root_dir, "test")
os.makedirs(out_dir, exist_ok=True)
target_imgs = [
    "x",
    "pred_seg",
    "pred_x_seg",
]

marge_images(target_dir, target_imgs, out_dir)