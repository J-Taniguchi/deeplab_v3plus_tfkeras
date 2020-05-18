import argparse
import json
import sys
import os

from glob import glob
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.append("../deeplab_v3plus_tfkeras")

from deeplab_v3plus_tfkeras.label import Label

parser = argparse.ArgumentParser()
parser.add_argument("label_list_path", help="path to label list csv file")
parser.add_argument("input_dir", help="path to input dir. all json files under this directory will be used to generate segmentation image file.")
parser.add_argument("output_dir", help="path to output dir. segmentation images will be generated to this directory.")

args = parser.parse_args()


def make_y_from_poly_json_path(data_path, palette):
    """Short summary.

    Args:
        data_path (path): path to json file.
        label (Label): class "Label" written in label.py

    Returns:
        np.array: y

    """

    with open(data_path) as d:
        poly_json = json.load(d)
    org_image_size = (poly_json["imageWidth"], poly_json["imageHeight"])
    n_poly = len(poly_json['shapes'])

    out_image = Image.new("P", size=org_image_size, color=0)
    out_image.putpalette(palette)

    draw = ImageDraw.Draw(out_image)

    for i in range(n_poly):
        label_name = poly_json['shapes'][i]['label']
        label_num = label.name.index(label_name)

        poly = poly_json['shapes'][i]['points']
        poly = tuple(map(tuple, poly))
        draw.polygon(poly, fill=label_num)

    return out_image


label_path = args.label_list_path
input_dir = args.input_dir
output_dir = args.output_dir

if os.path.exists(output_dir):
    tmp = input("Is it OK to write files to {}? ['yes' or 'no']: ".format(output_dir))
    if tmp != "yes":
        exit(0)
else:
    tmp = input("{} is not exist. Is it OK to make dir? ['yes' or 'no']: ".format(output_dir))
    if tmp == "yes":
        print("maked {}".format(output_dir))
        os.makedirs(output_dir)
    else:
        print("EXIT")
        exit(0)

label = Label(label_path)
palette = label.color.flatten().tolist()
data_paths = glob(os.path.join(input_dir, "*.json"))
print(data_paths)

for data_path in tqdm(data_paths):
    img = make_y_from_poly_json_path(data_path, palette)

    out_name = os.path.basename(data_path)
    out_name = os.path.splitext(out_name)[0] + ".png"
    img.save(os.path.join(output_dir, out_name))

print("end process")