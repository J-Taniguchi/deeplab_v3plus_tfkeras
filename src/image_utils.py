import numpy as np
import PIL.Image as Image

def make_x_from_image_paths(img_paths, image_size):
    n_images = len(img_paths)
    x = []
    for i in range(n_images):
        image = read_image(img_paths[i], image_size)
        image = np.array(image, np.uint8)
        x.append(image)
    return np.array(x)

def make_y_from_image_paths(img_paths, image_size, n_categories):
    n_images = len(img_paths)
    y = []
    for i in range(n_images):
        image = read_image(img_paths[i], image_size)
        image = np.array(image, np.int32)
        y0 = convert_index_png_array_to_y(image, n_categories)
        y.append(y0)
    return np.array(y)

def convert_index_png_array_to_y(index_png_image_array, n_categories):
    y = np.zeros((*(index_png_image_array).shape,n_categories), np.float32)
    for i in range(n_categories):
        if i == 0:
            y[:,:,i] = (index_png_image_array == 0) + (index_png_image_array==255)
        else:
            y[:,:,i] = (index_png_image_array==i)
    return y

def read_image(img_path, out_img_size):
    image = Image.open(img_path)
    image = image.resize(out_img_size)
    return image

def convert_y_to_image_array(y, image_size, n_categories, palette, threshold=0.0):
    out_img = []
    for i in range(y.shape[0]):
        out_img0 = np.zeros((*image_size, 3), np.float32)
        under_threshold = y[i,:,:,:].max(2) < threshold
        y[i,under_threshold,0] = 0.0
        max_category = y[i,:,:,:].argmax(2)
        for j in range(n_categories):
            out_img0[max_category==j] = palette[j,:]
        out_img.append(out_img0)
    return out_img

def make_palette(sample_image_path):
    sample_image = Image.open(sample_image_path)
    palette = sample_image.getpalette()
    palette = np.array(palette).reshape(-1, 3)/255
    return palette
