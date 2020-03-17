import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
matplotlib.use('Agg')

def visualise_true_pred(i,
                        x,
                        y_true,
                        y_pred,
                        out_dir,
                        last_activation,
                        label=None):
    if (label is None) and (last_activation == "sigmoid"):
        raise Exception("label is needed, when last_activation is sigmoid.")

    if last_activation == "softmax":
        fname = os.path.join(out_dir, "{:06}_pred_seg.png".format(i))
        cv2.imwrite(fname, y_pred[i][:,:,::-1])
        fname = os.path.join(out_dir, "{:06}_true_seg.png".format(i))
        cv2.imwrite(fname, y_true[i][:,:,::-1])
        fname = os.path.join(out_dir, "{:06}_x.png".format(i))
        cv2.imwrite(fname, x[i,:,:,::-1])

        y_mask = y_pred[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        x_mask = y_mask * x[i,:,:,:]
        fname = os.path.join(out_dir, "{:06}_pred_x_seg.png".format(i))
        cv2.imwrite(fname, x_mask[:,:,::-1].astype(np.uint8))

        y_mask = y_true[i].copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        x_mask = y_mask * x[i,:,:,:]
        fname = os.path.join(out_dir, "{:06}_true_x_seg.png".format(i))
        cv2.imwrite(fname, x_mask[:,:,::-1].astype(np.uint8))

    elif last_activation == "sigmoid":
        fname = os.path.join(out_dir, "{:06}_x.png".format(i))
        cv2.imwrite(fname, x[i,:,:,::-1])

        for j in range(label.n_labels):
            label_name =label.name[j]

            fname = os.path.join(
                out_dir, "{:06}_{}_pred_seg.png".format(i, label_name))
            cv2.imwrite(fname, y_pred[i][j][:,:,::-1])
            fname = os.path.join(
                out_dir, "{:06}_{}_true_seg.png".format(i, label_name))
            cv2.imwrite(fname, y_true[i][j][:,:,::-1])

            y_mask = y_pred[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            fname = os.path.join(
                out_dir, "{:06}_{}_pred_x_seg.png".format(i, label_name))
            x_mask = (y_mask * x[i,:,:,:]).astype(np.uint8)
            cv2.imwrite(fname, x_mask[:,:,::-1])

            y_mask = y_true[i][j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            fname = os.path.join(
                out_dir, "{:06}_{}_true_x_seg.png".format(i, label_name))
            x_mask = (y_mask * x[i,:,:,:]).astype(np.uint8)
            cv2.imwrite(fname, x_mask[:,:,::-1])

def visualise_pred(i,
                   x,
                   y,
                   out_dir,
                   last_activation,
                   label=None):
    if (label is None) and (last_activation == "sigmoid"):
        raise Exception("label is needed, when last_activation is sigmoid.")
    now_x = x[i]
    now_y = y[i]

    if last_activation == "softmax":
        fname = os.path.join(out_dir, "{:06}_seg.png".format(i))
        cv2.imwrite(fname, now_y[:,:,::-1])
        fname = os.path.join(out_dir, "{:06}_x.png".format(i))
        cv2.imwrite(fname, now_x[:,:,::-1])

        y_mask = now_y.copy()/255
        black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
        y_mask[black_pix,:] = [1.0,1.0,1.0]
        fname = os.path.join(out_dir, "{:06}_x_seg.png".format(i))
        cv2.imwrite(fname, (y_mask*now_x).astype(np.uint8))

    elif last_activation == "sigmoid":
        for j in range(label.n_labels):
            label_name =label.name[j]
            fname = os.path.join(
                out_dir, "{:06}_{}_seg.png".format(i, label_name))
            cv2.imwrite(fname, now_y[j][:,:,::-1])

            if j == 0:
                fname = os.path.join(out_dir, "{:06}_x.png".format(i))
                cv2.imwrite(fname, now_x[:,:,::-1])

            y_mask = now_y[j].copy()/255
            black_pix=(y_mask == np.array([0.0,0.0,0.0])).all(axis=2)
            white_pix=(y_mask == np.array([1.0,1.0,1.0])).all(axis=2)
            y_mask[black_pix,:] = [0.5,0.5,0.5]
            y_mask[white_pix,:] = [0.5,0.5,0.5]
            fname = os.path.join(
                out_dir, "{:06}_{}_x_seg.png".format(i, label_name))
            x_mask = (y_mask*now_x).astype(np.uint8)
            cv2.imwrite(fname, x_mask[:,:,::-1])


def convert_y_to_image_array(y, label, threshold=0.5, activation="softmax"):
    out_img = []
    # y is list or 4D np.array.(batch,h,w,class)
    for i in range(len(y)):
        now_img = y[i]
        if activation == "softmax":
            out_img0 = np.zeros((now_img.shape[0],
                                 now_img.shape[1],
                                 3), np.uint8)
            under_threshold = now_img.max(2) < threshold
            now_img[i,under_threshold,0] = 1.0
            max_category = now_img.argmax(2)
            for j in range(label.n_labels):
                out_img0[max_category==j] = label.color[j,:]
            out_img.append(out_img0)
        elif activation == "sigmoid":
            tmp = []
            for j in range(label.n_labels):
                out_img0 = np.zeros((now_img.shape[0],
                                     now_img.shape[1],
                                     3), np.uint8)
                tar_idx = now_img[:,:,j] > threshold
                out_img0[tar_idx,:] = label.color[j,:]
                tmp.append(out_img0)
            out_img.append(tmp)
        else:
            print("activation is " + activation)
            raise Exception("activation must be 'softmax' or 'sigmoid'")
    return out_img
