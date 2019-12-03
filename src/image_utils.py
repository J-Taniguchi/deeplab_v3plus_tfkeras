import numpy as np
import PIL.Image as Image
import pandas as pd

def make_x_from_image_paths(img_paths, image_size):
    n_images = len(img_paths)
    x = []
    for i in range(n_images):
        image = read_image(img_paths[i], image_size)
        image = np.array(image, np.uint8)
        x.append(image)
    return np.array(x)

def make_y_from_image_paths(img_paths, image_size, label_np, is_index_png=False):
    n_categories = label_np.shape[0]
    n_images = len(img_paths)
    y = []
    for i in range(n_images):
        image = read_image(img_paths[i], image_size)
        if is_index_png:
            image = image.convert('RGB')
        image = np.array(image, np.int32)[:,:,:3]
        y0 = convert_image_array_to_y(image, label_np)
        y.append(y0)
    return np.array(y)

def convert_image_array_to_y(image_array, label_np):
    n_categories = label_np.shape[0]
    y = np.zeros((*(image_array).shape[:2], n_categories), np.float32)
    for i in range(n_categories):
        y[:,:,i] = np.all(np.equal(image_array , label_np[i,:]), axis=2).astype(np.float32)
    return y

def read_image(img_path, out_img_size):
    image = Image.open(img_path)
    image = image.resize(out_img_size)
    return image
#要動作確認
def convert_y_to_image_array(y, image_size, label_np, threshold=0.0):
    out_img = []
    for i in range(y.shape[0]):
        out_img0 = np.zeros((*image_size, 3), np.float32)
        under_threshold = y[i,:,:,:].max(2) < threshold
        y[i,under_threshold,0] = 0.0
        max_category = y[i,:,:,:].argmax(2)
        for j in range(n_categories):
            out_img0[max_category==j] = label_np[j,:]
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
