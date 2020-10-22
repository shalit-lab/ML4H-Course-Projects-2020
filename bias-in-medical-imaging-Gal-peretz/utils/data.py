import math
import urllib.request
import os
import re
import numpy as np
from PIL import ImageDraw,ImageDraw2
from PIL.ImageFont import ImageFont
from tqdm import tqdm
import zipfile
import hdf5storage


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, target_folder, filename):
    # check if data exists
    print("Check if data exists on disk")
    if not os.path.isdir(target_folder):
      print("Creating target folder")
      os.mkdir(target_folder)
    files = os.listdir(target_folder)
    if not files:
        print("Cannot find files on disk")
        print("Downloading files")
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=target_folder + filename, reporthook=t.update_to)
    print("Download completed!")

def unzip_all_files(target_folder):
    print("Unzip files")
    items = os.listdir(target_folder)
    while(any(item.endswith('.zip') for item in items)):
        for item in filter(lambda item: item.endswith('.zip'), items):
            with zipfile.ZipFile(target_folder + item, "r") as zip_ref:
                zip_ref.extractall(target_folder)
        for item in items:
            if item.endswith(".zip"):
                os.remove(target_folder + item)
        items = os.listdir(target_folder)
    print("Unzip completed!")

def convert_landmark_to_bounding_box(landmark):
    x_min = x_max = y_min = y_max = None
    for x, y in landmark:
        if x_min is None:
            x_min = x_max = x
            y_min = y_max = y
        else:
            x_min, x_max = min(x, x_min), max(x, x_max)
            y_min, y_max = min(y, y_min), max(y, y_max)
    return [int(x_min), int(x_max), int(y_min), int(y_max)]

def distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


def get_extremes_points(landmark):
    x_min = x_max = y_min = y_max = None
    left_most = right_most = top_most = bottom_most = None
    for x, y in landmark:
        if x_min is None:
            x_min = x_max = x
            y_min = y_max = y
            left_most = right_most = top_most = bottom_most = (x, y)
        else:
            if x < x_min:
                left_most = (x, y)
                x_min = x
            if y < y_min:
                top_most = (x, y)
                y_min = y
            if x > x_max:
                right_most = (x, y)
                x_max = x
            if y > y_max:
                bottom_most = (x, y)
                y_max = y
    return left_most, right_most, top_most, bottom_most


def convert_landmark_to_line_coordination(landmark):
    extremes_point = get_extremes_points(landmark)
    pt1 = pt2 = (0,0)
    for p1 in extremes_point:
        for p2 in extremes_point:
            if distance(pt1,pt2) < distance(p1,p2):
                pt1 = p1
                pt2 = p2
    return (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))

def linedashed(draw, pt1, pt2, dashlen=4, ratio=2):
    x0, y0 = pt1
    x1, y1 = pt2
    dx = x1 - x0
    dy = y1 - y0
    if dy == 0:
        len = dx
    elif dx == 0:
        len = dy
    else:
        len = math.sqrt(dx * dx + dy * dy)
    xa = dx / len
    ya = dy / len
    step = dashlen * ratio
    a0 = 0
    while a0 < len:
        a1 = a0 + dashlen
        if a1 > len: a1 = len
        draw.line((x0 + xa * a0, y0 + ya * a0, x0 + xa * a1, y0 + ya * a1), fill="red")
        a0 += step


def add_distance_label(draw, pt1, pt2):
    x0, y0 = pt1
    x1, y1 = pt2
    offset = 5
    x_half = min(x0, x1) + (abs(x0 - x1) // 2)
    y_half = min(y0, y1) + (abs(y0 - y1) // 2)
    x_half += offset
    y_half += offset
    draw.text((x_half, y_half), text=f"{round(distance(pt1, pt2)/20, 2)}cm", fill="red")

def draw_measurement_line(img, landmarks):
    pt1, pt2 = convert_landmark_to_line_coordination(landmarks)
    draw = ImageDraw.Draw(img)
    linedashed(draw, pt1, pt2)
    add_distance_label(draw, pt1, pt2)
    return img

def draw_area_annotations(img, landmarks):
    draw = ImageDraw2.Draw(img)
    pen = ImageDraw2.Pen("blue", width=50)
    draw.polygon(landmarks, pen)
    extremes_points = get_extremes_points(landmarks)
    right_most = extremes_points[1]
    pt1, pt2 = convert_landmark_to_line_coordination(landmarks)
    draw = ImageDraw.Draw(img)
    draw.text((right_most[0]+20, right_most[1]), text=f"{round(distance(pt1, pt2)/10, 2)}cm", fill="blue")
    return img


def _arrange_brain_tumor_data(root):
    """
        -----
    This data is organized in matlab data format (.mat file). Each file stores a struct
    containing the following fields for an image:

    cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
    cjdata.PID: patient ID
    cjdata.image: image data
    cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.
            For example, [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on tumor border.
            It was generated by manually delineating the tumor border. So we can use it to generate
            binary image of tumor mask.
    cjdata.tumorMask: a binary image with 1s indicating tumor region

    -----
    """
    # Remove and split files
    items = [item for item in filter(lambda item: re.search("^[0-9]+\.mat$", item), os.listdir(root))]
    try:
        os.mkdir(root + 'meningioma/')
    except:
        print("Meningioma directory already exists")
    try:
        os.mkdir(root + 'glioma/')
    except:
      print("Glioma directory already exists")
    try:
        os.mkdir(root + 'pituitary/')
    except:
        print("Pituitary directory already exists")

    for item in items:
        sample = hdf5storage.loadmat(root + item)['cjdata'][0]
        if sample[2].shape[0] == 512:
            label = sample[0].item()
            label = int(label)
            if label == 1:
                os.rename(root + item, root + 'meningioma/' + item)
            elif label == 2:
                os.rename(root + item, root + 'glioma/' + item)
            elif label == 3:
                os.rename(root + item, root + 'pituitary/' + item)
        else:
            os.remove(root + item)

def get_data_if_needed(data_path='./data/', url="https://ndownloader.figshare.com/articles/1512427/versions/5"):
    if os.path.isdir(data_path):
        print("Data directory already exists. ",
              "if from some reason the data directory structure is wrong please remove the data dir and rerun this script")
        return
    filename = "all_data.zip"
    download_url(url, data_path, filename)
    unzip_all_files(data_path)
    _arrange_brain_tumor_data(data_path)