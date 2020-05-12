import numpy as np
import h5py
import os
import sys
from PIL import Image
from tqdm import tqdm
from glob import glob
deeplab_v3plus_tfkeras_dir = "./"
sys.path.append(deeplab_v3plus_tfkeras_dir)

from deeplab_v3plus_tfkeras.data_utils import convert_image_array_to_y, make_y_from_poly_json_path
from deeplab_v3plus_tfkeras.label import Label

out_name = "../../data/train_data_512.h5"
train_dir = "../../data/train_data"


def make_cut_xy_h5(x_path, y_path, data_type, out_image_size, label, out_dir=None):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    image_name = os.path.basename(x_path).split(".")[0]
    x0 = Image.open(x_path)
    x0 = np.array(x0)[:, :, :3]
    h_org, w_org = x0.shape[0:2]
    w = w_org + (out_image_size[1] - 1) * 2
    h = h_org + (out_image_size[0] - 1) * 2
    image_size_quarter = (out_image_size[0] // 4, out_image_size[1] // 4)
    x_inds = np.arange(0, w, image_size_quarter[0])[0:-1]
    y_inds = np.arange(0, h, image_size_quarter[1])[0:-1]

    # read y
    if data_type == "image":
        image = Image.open(y_path)
        image = np.array(image, np.int32)[:, :, :3]
        y0 = convert_image_array_to_y(image, label)
    elif data_type == "index_png":
        image = Image.open(y_path)
        image = image.convert('RGB')
        image = np.array(image, np.int32)[:, :, :3]
        y0 = convert_image_array_to_y(image, label)
    elif data_type == "polygon":
        y0 = make_y_from_poly_json_path(y_path, (w_org, h_org), label)
    else:
        raise Exception("data_type must be \"image\" or \"index_png\" or \"polygon\".")

    padded_x = np.zeros((h, w, 3), dtype=np.uint8)
    padded_y = np.zeros((h, w, label.n_labels), dtype=np.uint8)
    # put original image to larger black image.
    padded_x[(out_image_size[1] - 1):(out_image_size[1] - 1) + h_org,
             (out_image_size[0] - 1):(out_image_size[0] - 1) + w_org, :] = x0
    padded_y[(out_image_size[1] - 1):(out_image_size[1] - 1) + h_org,
             (out_image_size[0] - 1):(out_image_size[0] - 1) + w_org, :] = y0

    x = []
    y = []
    names = []
    for i, x_ind in enumerate(x_inds):
        for j, y_ind in enumerate(y_inds):
            croped_x = padded_x[y_ind:y_ind + out_image_size[1], x_ind:x_ind + out_image_size[0], :]
            croped_y = padded_y[y_ind:y_ind + out_image_size[1], x_ind:x_ind + out_image_size[0], :]
            out_x = np.zeros((out_image_size[0], out_image_size[1], 3), dtype=np.uint8)
            out_y = np.zeros((out_image_size[0], out_image_size[1], label.n_labels), dtype=np.uint8)
            out_x[0:croped_x.shape[0], 0:croped_x.shape[1]] = croped_x.copy()
            out_y[0:croped_y.shape[0], 0:croped_y.shape[1]] = croped_y.copy()
            if (out_x == 0).sum() / 3 > (out_image_size[0] * out_image_size[1] / 2):
                continue
            x.append(out_x)
            y.append(out_y)
            name = image_name + "_{}_{}".format(i, j)
            names.append(name)
            if out_dir is not None:
                Image.fromarray(out_x).save(os.path.join(out_dir, name + ".png"))
    return x, y, names


x_paths = glob(os.path.join(train_dir, "*.png"))
x_paths.sort()
y_paths = glob(os.path.join(train_dir, "*.json"))
y_paths.sort()

label_file_path = os.path.join(train_dir, 'label.csv')
label = Label(label_file_path)

n_data = len(x_paths)
x = []
y = []
names = []
for i in tqdm(range(n_data)):
    x_path = x_paths[i]
    y_path = y_paths[i]
    _x, _y, _names = make_cut_xy_h5(x_path,
                                    y_path,
                                    data_type="polygon",
                                    out_image_size=(512, 512),
                                    label=label,
                                    # out_dir="../../data/train_data_256"
                                    )
    x.extend(_x)
    y.extend(_y)
    names.extend(_names)

x = np.array(x)
y = np.array(y)
names = np.array(names, dtype=h5py.special_dtype(vlen=str))
with h5py.File(out_name, "w") as f:
    f["x"] = x
    f["y"] = y
    f["names"] = names
