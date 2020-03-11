import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from glob import glob
import os
import sys
from deeplab_v3plus_tfkeras.data_utils import make_xy_from_data_paths, convert_y_to_image_array, inference_large_img, save_inference_results
from deeplab_v3plus_tfkeras.data_gen import DataGenerator
from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.metrics import make_IoU
#from deeplab_v3plus_tfkeras.loss import make_overwrap_crossentropy
from deeplab_v3plus_tfkeras.loss import make_weighted_overwrap_crossentropy
#from deeplab_v3plus_tfkeras.loss import make_overwrap_focalloss
from tensorflow.keras.utils import get_custom_objects
from tqdm import tqdm

#model_dir = "../deeplab_out/add_no5data_OFL_decay"
#traindata_dir = '../../data/train_data'
#validdata_dir = '../../data/'
model_dir = sys.argv[1]
traindata_dir = sys.argv[2]
testdata_dirs = sys.argv[3:]

image_size = (256,256)

gpu_options = tf.compat.v1.GPUOptions(visible_device_list="3", allow_growth=False)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


label_file_path = os.path.join(traindata_dir, 'label.csv')
label = Label(label_file_path)
get_custom_objects()["IoU"] = make_IoU(threshold=0.5)
#get_custom_objects()["overwrap_crossentropy"] = make_overwrap_crossentropy(label.n_labels)
weights = [[0.996, 0.004], [0.996, 0.004]]
get_custom_objects()["weighted_overwrap_crossentropy"] = \
    make_weighted_overwrap_crossentropy(label.n_labels, weights)

#get_custom_objects()["overwrap_focalloss"] = \
    #make_overwrap_focalloss(label.n_labels)

model = keras.models.load_model(os.path.join(model_dir,'best_model.h5'))
preprocess = keras.applications.xception.preprocess_input

last_activation = model.layers[-1].name

train_x_paths = glob(os.path.join(traindata_dir,'*.png'))
train_x_paths.sort()
image_names = [os.path.basename(train_x_paths[i]).split('.')[0] for i in range(len(train_x_paths))]
train_y_paths=[]
for i, image_name in enumerate(image_names):
    p = os.path.join(traindata_dir, image_name+'.json')
    if os.path.exists(p):
        train_y_paths.append(p)
    else:
        train_y_paths.append(None)



valid_x, valid_y = make_xy_from_data_paths(train_x_paths,
                                           train_y_paths,
                                           image_size,
                                           label,
                                           "polygon",
                                           resize_or_crop="crop")

pred = model.predict(preprocess(valid_x), batch_size=8)
fpath = os.path.join(model_dir, "trained.h5")
save_inference_results(fpath,
                       x=valid_x,
                       pred=pred,
                       y=valid_y,
                       last_activation=last_activation)

for testdata_dir in testdata_dirs:
    test_name = testdata_dir.split(os.sep)[-1]
    test_x_paths = glob(os.path.join(testdata_dir,'*.png'))
    test_x_paths.sort()

    tar = range(len(test_x_paths))

    mode = "max_confidence"
    x_imgs = []
    seg_imgs =[]

    for i in tqdm(range(len(test_x_paths))):
    #for i in tqdm(np.arange(0, len(valid_x_paths), 1000)):
        x_img, seg_img = inference_large_img(
            test_x_paths[i],
            model,
            preprocess,
            label,
            mode=mode,
            threshold=0.5)
        x_imgs.append(x_img)
        seg_imgs.append(seg_img)

    x_imgs = np.array(x_imgs)
    seg_imgs = np.array(seg_imgs)

    fpath = os.path.join(model_dir, test_name + ".h5")
    save_inference_results(fpath,
                           x=x_imgs,
                           pred=seg_imgs,
                           last_activation=last_activation)
