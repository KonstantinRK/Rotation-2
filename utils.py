import os
import shutil
import numpy as np
import cv2
import random
import math
import re
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


_COL_MAP = {"red": [1, 0, 0],
            "green": [0, 1, 0],
            "blue": [0, 0, 1],
            "aqua": [0, 1, 1],
            "pink": [1, 0, 1],
            "yellow": [1, 1, 0],
            "white": [1, 1, 1]}


_COL_MAP_INV = {"blue": [1, 0, 0],
                "green": [0, 1, 0],
                "red": [0, 0, 1],
                "yellow": [0, 1, 1],
                "pink": [1, 0, 1],
                "aqua": [1, 1, 0],
                "white": [1, 1, 1]}


def color_bw_image(img, color, inv_map=False):
    cmap = _COL_MAP_INV if inv_map else _COL_MAP
    base = [img] * 3
    d = [base[k] * cmap[color][k] for k in range(3)]
    return np.stack(d, axis=2)


def change_hue_rgb_image(img, color, inv_map=True):
    cmap = _COL_MAP_INV if inv_map else _COL_MAP
    for i, c in enumerate(cmap[color]):
        if c == 0:
            img[:, :, i] = 0
    return img


def create_circle_img(shape, color="white", thickness=None, ):
    img = np.zeros(shape)
    thickness = random.choice([-1] + list(range(1, int((min(shape) * 0.2))))) if thickness is None else thickness
    border = (shape[0] - thickness, shape[1] - thickness)
    size = random.randint(int((min(border) // 2) * 0.1), int((min(border) // 2) * 0.9))
    offset = size+thickness
    try:
        pos = (random.randint(offset, border[1] - offset), random.randint(offset, border[0] - offset))
        img = cv2.circle(img, pos, size, (255, 255, 255), thickness=thickness)
        img = color_bw_image(img, color, inv_map=False)
        # cv2.imshow("name", img)
        # cv2.waitKey(0)
        return img
    except ValueError:
        return create_circle_img(shape, color)


def create_rectangle_img(shape, color="white", thickness=None):
    img = np.zeros(shape)
    thickness = random.choice([-1] + list(range(1, int((min(shape) * 0.2))))) if thickness is None else thickness
    border = (shape[0] - thickness, shape[1] - thickness)
    size_x = random.randint(int(border[1] * 0.1), int(border[1] * 0.9))
    size_y = random.randint(int(border[0] * 0.1), int(border[0] * 0.9))
    pos_1 = (random.randint(0, border[1] - size_x), random.randint(0, border[0] - size_y))
    pos_2 = (pos_1[0]+size_x, pos_1[1]+size_y)
    try:
        img = cv2.rectangle(img, pos_1, pos_2 , (255, 255, 255), thickness=thickness)
        img = color_bw_image(img, color, inv_map=False)
        # cv2.imshow("name", img)
        # cv2.waitKey(0)
        return img
    except ValueError:
        return create_rectangle_img(shape, color)


def create_triangle_img(shape, color="white", thickness=None):
    img = np.zeros(shape)
    thickness = random.choice([-1] + list(range(1, int((min(shape) * 0.2))))) if thickness is None else thickness
    border = (shape[0] - thickness, shape[1] - thickness)
    size_x = random.randint(int(border[1] * 0.1), int(border[1] * 0.9))
    size_y = random.randint(int(border[0] * 0.1), int(border[0] * 0.9))
    pos_1 = (random.randint(0, border[1] - size_x), random.randint(0, border[0] - size_y))
    pos_2 = (pos_1[0]+size_x, pos_1[1]+random.randint(0, size_y))
    pos_3 = (pos_1[0]+random.randint(0, size_x), pos_1[1]+size_y)
    try:
        img = cv2.drawContours(img, [np.array([pos_1, pos_2, pos_3])], 0, (255, 255, 255), thickness=thickness)
        img = color_bw_image(img, color, inv_map=False)
        # cv2.imshow("name", img)
        # cv2.waitKey(0)
        return img
    except ValueError:
        return create_rectangle_img(shape, color)


def rotate(origin, point, angle, as_int=True):
    angle = -  math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    if as_int:
        qx = int(round(qx, 0))
        qy = int(round(qy, 0))
    return qx, qy


def create_line_img(shape, color="white", degree=None):
    if degree is None:
        degree = random.randint(0, 360)
    img = np.zeros(shape)
    thickness = random.choice(list(range(1, int((min(shape)*0.2)))))
    size = random.randint(int(min(shape)*0.1), int(min(shape)*0.9))
    offset = int(size*0.25)
    pos_1 = (random.randint(offset, shape[1]-offset), random.randint(offset, shape[0]-offset))
    pos_2 = rotate(pos_1, (pos_1[0]+size, pos_1[1]), degree)
    try:
        img = cv2.line(img, pos_1, pos_2, (255, 255, 255), thickness=thickness)
        img = color_bw_image(img, color, inv_map=False)
        return img
    except ValueError:
        return create_line_img(shape, color, degree)


def create_crescent_img(shape, color="white"):
    img = np.zeros(shape)
    thickness = random.choice(list(range(1, int((min(shape) * 0.2)))))
    border = (shape[0] - thickness, shape[1] - thickness)
    size = random.randint(int((min(border) // 2) * 0.1), int((min(border) // 2) * 0.9))
    offset = size+thickness
    angle = random.randint(0, 360)
    start_angle = 90
    end_angle = random.randint(start_angle+90, 360)
    try:
        pos = (random.randint(offset, border[1] - offset), random.randint(offset, border[0] - offset))
        img = cv2.ellipse(img, pos, (size, size), angle, start_angle, end_angle, (255, 255, 255), thickness=thickness)
        img = color_bw_image(img, color, inv_map=False)
        return img
    except ValueError:
        return create_crescent_img(shape, color)


class BaseClass(object):

    _NAME = "bc"

    def __init__(self, root="", *args, **kwargs):
        self.root_dir = root

    def get_name(self):
        return self._NAME

    def get_path(self, file_name=None):
        if file_name is None:
            return os.path.join(self.root_dir, self.get_name())
        elif isinstance(file_name, list):
            return os.path.join(self.root_dir, self.get_name(), *file_name)
        else:
            return os.path.join(self.root_dir, self.get_name(), file_name)

    def _setup(self):
        os.makedirs(self.get_path(), exist_ok=True)

    def setup(self):
        # print(self.get_path())
        self._setup()

    def delete(self, file_name=None):
        path = self.get_path(file_name)
        if os.path.exists(path):
            shutil.rmtree(path)


def translate_output(name, as_pandas=True):
    path = "outputs"
    with open(os.path.join(path, "input", name), "r") as f:
        txt_raw = f.read()
    classes = re.findall("Class (\d+)", txt_raw)
    layers = re.findall("Layer (\d+)", txt_raw)
    concepts = re.findall("Concept = (.*)", txt_raw)
    bottlenecks = re.findall("Bottleneck = (.+?)\.", txt_raw)
    tcav = re.findall("TCAV Score = (.+?)\s+?\(", txt_raw)
    tcav_pm = re.findall("TCAV Score = .+?\s+?\(\+-\s+?(.+?)\)", txt_raw)
    rand = re.findall("random was\s+?(.+?)\s+?\(", txt_raw)
    rand_pm = re.findall("random was\s+?.+?\s+?\(\+-\s+?(.+?)\)", txt_raw)
    p_val = re.findall("p-val = (.+?)\s+?\(", txt_raw)
    significant = re.findall("p-val = .+?\s+?\((.+?)\)", txt_raw)

    data = []
    cl = -1
    for i in range(len(significant)):
        if i % (len(significant) / len(classes)) == 0:
            cl += 1
        sig = bool(significant[i] == "significant")
        row = [classes[cl],
               bottlenecks[i],
               concepts[i],
               tcav[i],
               tcav_pm[i],
               rand[i],
               rand_pm[i],
               p_val[i],
               sig]
        data.append(row)
    if as_pandas:
        df = pd.DataFrame(data, columns=["label", "layer", "concept", "tcav", "tcav_pm", "rand", "rand_pm", "p_val", "sig"])
        df.to_csv(os.path.join(path, "output", os.path.splitext(name)[0]+".csv"))
        return df
    else:
        return data


def read_output(name):
    return pd.read_csv(os.path.join("outputs", "output", name))


def list_dict(inp_dict):
    result = set()
    for k, v in inp_dict.items():
        result.add(k)
        result = result.union(list_dict(v))
    return result


def find_cutoff(data):
    vals = np.sort(np.array(data))
    a = vals.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(a)
    s = np.linspace(0, 1, num=100)
    e = kde.score_samples(s.reshape(-1, 1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    cutoff = s[mi[-1]]
    return cutoff