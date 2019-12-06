# deeplab_v3plus_tfkeras

Programs for semantic segmentation. Useage is written in ipynb files.

- you can annotate as
  - RGB image
  - index png image(like pascal voc2012)
  - polygon ([labelme](https://github.com/wkentaro/labelme) json file.)
- It is better to modify data_augment.py for your objective.
  - [albumentations](https://github.com/albumentations-team/albumentations) is used for data augment.
