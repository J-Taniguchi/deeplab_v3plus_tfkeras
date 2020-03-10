import numpy as np
import os
import sys
import joblib
n_jobs=36
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.data_utils import load_inference_results, convert_y_to_image_array
import h5py

#model_dir = "../deeplab_out/add_no5data_OFL_decay"
#traindata_dir = '../../data/train_data'
#validdata_dir = '../../data/'
model_dir = sys.argv[1]
traindata_dir = sys.argv[2]
validdata_dir = sys.argv[3]
#valid_names = ["valid_4-09", "valid_4-10"]
valid_names = ["valid_4-09"]


out_dir = os.path.join(model_dir,"figure")

label_file_path = os.path.join(traindata_dir, 'label.csv')
label = Label(label_file_path)

matplotlib.use('Agg')

def visualise_true_pred(i, x, y_true, y_pred, last_activation):
    if last_activation == "softmax":
        img = Image.fromarray(y_pred[i])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_pred_seg.png"))
        img = Image.fromarray(y_true[i])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_true_seg.png"))

        img = Image.fromarray(x[i,:,:,:])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_x.png"))

        y_mask = y_pred[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        img = Image.fromarray((y_mask*x[i,:,:,:]).astype(np.uint8))
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_pred_x_seg.png"))

        y_mask = y_true[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        img = Image.fromarray((y_mask*x[i,:,:,:]).astype(np.uint8))
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_true_x_seg.png"))
    elif last_activation == "sigmoid":
        img = Image.fromarray(x[i,:,:,:])
        img.save(os.path.join(out_dir_train,str(i).zfill(6) + "_x.png"))
        for j in range(label.n_labels):
            label_name =label.name[j]

            img = Image.fromarray(y_pred[i][j])
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_pred_seg.png"))
            img = Image.fromarray(y_true[i][j])
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_true_seg.png"))

            y_mask = y_pred[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            img = Image.fromarray((y_mask*x[i,:,:,:]).astype(np.uint8))
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_pred_x_seg.png"))

            y_mask = y_true[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            img = Image.fromarray((y_mask*x[i,:,:,:]).astype(np.uint8))
            img.save(os.path.join(out_dir_train,str(i).zfill(6) + label_name + "_true_x_seg.png"))


def visualise_pred(i, x, y, last_activation, out_dir_valid):
    if last_activation == "softmax":
        img = Image.fromarray(y)
        img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_seg.png"))

        img = Image.fromarray(x)
        img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_x.png"))

        y_mask = y.copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        img = Image.fromarray((y_mask*x).astype(np.uint8))
        img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_x_seg.png"))
    elif last_activation == "sigmoid":
        for j in range(label.n_labels):
            label_name =label.name[j]

            img = Image.fromarray(y[i,j,:,:,:])
            img.save(os.path.join(out_dir_valid, str(i).zfill(6) + label_name + "_seg.png"))

            if j == 0:
                img = Image.fromarray(x[i,:,:,:])
                img.save(os.path.join(out_dir_valid, str(i).zfill(6) + "_x.png"))

            y_mask = y[i,j,:,:,:].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            img = Image.fromarray((y_mask*x[i,:,:,:]).astype(np.uint8))
            img.save(os.path.join(out_dir_valid, str(i).zfill(6) + label_name + "_x_seg.png"))

#trained data
'''
out_dir_train = os.path.join(out_dir, "train")
os.makedirs(out_dir_train,exist_ok = True)

fpath = os.path.join(model_dir, "trained.h5")
x, y, pred, last_activation = load_inference_results(fpath)

y_pred = convert_y_to_image_array(pred, label, threshold=0.5, activation=last_activation)
y_true = convert_y_to_image_array(y, label, activation=last_activation)

joblib.Parallel(n_jobs=n_jobs, verbose=10, backend="threading")(joblib.delayed(visualise_true_pred)(i, x, y_true, y_pred, last_activation) for i in range(x.shape[0]))
'''
#valid data
for valid_name in valid_names:
    print(valid_name)
    out_dir_valid = os.path.join(out_dir, valid_name)
    os.makedirs(out_dir_valid, exist_ok = True)

    fpath = os.path.join(model_dir, valid_name + ".h5")
    x, y, pred, last_activation = load_inference_results(fpath)

    #y_pred = convert_y_to_image_array(pred, label, threshold=0.5, activation=last_activation)

    joblib.Parallel(n_jobs=n_jobs, verbose=10, backend="threading")(joblib.delayed(visualise_pred)(i, x, pred, last_activation, out_dir_valid) for i in range(x.shape[0]))
