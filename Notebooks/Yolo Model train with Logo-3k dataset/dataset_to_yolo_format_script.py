import os
import random
import glob
import shutil

import json
import yaml

from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lxml import etree as ET

import pandas as pd
import matplotlib.pyplot as plt

import cv2
# from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from IPython.display import display, Image

from sklearn.model_selection import train_test_split
from IPython import display


dataset_dir = "/home/vihanga-ms/Desktop/research/archive/LogoDet-3K"
# dataset_dir = "/home/sinewave/logocode_detection/datasets/LogoDet-3K"
os.listdir(dataset_dir)

df = pd.DataFrame(glob.glob(f"{dataset_dir}/*/*/*"), columns=["file_path"])
df["ext"] = df["file_path"].apply(lambda x: x.split(".")[-1])
df["logo_category"] = df["file_path"].apply(lambda x: x.split(os.sep)[-3])
df["logo_name"] = df["file_path"].apply(lambda x: x.split(os.sep)[-2])
df.head()

df.to_csv("logodet3k_reference.csv", index=False)

df["ext"].value_counts()

df2 = df[df["ext"]=="jpg"].iloc[:]
print({
    "No. of categories": df2["logo_category"].nunique(), 
    "No. of logo types": df2["logo_name"].nunique(),
    "Avg. no. of images per category": df2.groupby("logo_category")["file_path"].count().mean().round(), 
    "Avg. no. of images per logo": df2.groupby("logo_name")["file_path"].count().mean().round(), 
})

df2["logo_category"].value_counts()

df2["logo_name"].value_counts().reset_index().plot(
    x="logo_name", y="count", figsize=(10,5), title="Distribution of logo img counts")

dataset_dst_dir = "/home/vihanga-ms/Desktop/research/archive"
if os.path.exists(dataset_dst_dir):
    shutil.rmtree(dataset_dst_dir)
os.makedirs(f"{dataset_dst_dir}/train", exist_ok=True)
os.makedirs(f"{dataset_dst_dir}/val", exist_ok=True)

classname2idx = {logo_name: idx for idx, logo_name in enumerate(sorted(df2["logo_name"].unique()))}
print(str(classname2idx)[:100]+"...")
idx2classname = {idx: logo_name for logo_name, idx in classname2idx.items()}

classname2idx = {"logo": 0}
idx2classname = defaultdict(lambda: "logo")

class_name_idx_map_str = "\n".join([f"    {idx}: {class_name}" for class_name, idx in classname2idx.items()])
print(class_name_idx_map_str)

dataset_config = f"""
path: {dataset_dst_dir} 
train:
    - train
val:
    - val

# test:
#     - test

# Classes
names:
{class_name_idx_map_str}
"""
print(dataset_config)
with open("dataset_config.yaml", "w") as f:
    f.write(dataset_config)
print("-"*10)
with open("dataset_config.yaml", "r") as f:
    datcon = yaml.safe_load(f)
    print(datcon)

def convert_voc_to_yolo(src, dst, classname2idx):
    tree = ET.parse(src)
    root = tree.getroot()
    yolo_lines = []
    image_width = float(root.find("size/width").text)
    image_height = float(root.find("size/height").text)
    depth = float(root.find("size/depth").text)
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        class_index = classname2idx.get(class_name, 0)
#         yolo_line = f"{class_index} {round(x_center, 6)} {round(y_center, 6)} {round(width, 6)} {round(height, 6)}"
        yolo_line = f"{class_index} {x_center} {y_center} {width} {height}"
        yolo_lines.append(yolo_line)
    if dst is not None:
        with open(dst, "w") as f:
            f.write("\n".join(yolo_lines))
    return yolo_lines
    
convert_voc_to_yolo(f"{dataset_dir}/Clothes/2xist/1.xml", None, {})

df2['is_train'] = True
train_df, test_df = train_test_split(df2, test_size=0.2, random_state=101)
test_df['is_train'] = False
final_df = pd.concat([train_df, test_df])
final_df.reset_index(drop=True, inplace=True)

def copy_to_working(x):
    train_folder = "train" if x["is_train"] else "val"
    src = x["file_path"]
    dst = os.path.join(dataset_dst_dir, train_folder, "__".join(x["file_path"].split(os.sep)[-3:]))
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    if not os.path.exists(dst.replace(".jpg", ".txt")):
        convert_voc_to_yolo(src.replace(".jpg", ".xml"), dst.replace(".jpg", ".txt"), classname2idx)
    return True

# copy_to_working(final_df.iloc[0].T.to_dict())
copy_to_working_results = []

with ThreadPoolExecutor() as e:
    for _, row in tqdm(final_df.iterrows()):
        status = e.submit(copy_to_working, dict(row))
        copy_to_working_results.append(status)
        
copy_to_working_results = final_df.apply(lambda x: copy_to_working(x), axis=1)
copy_to_working_results.sum(), final_df.shape[0]
