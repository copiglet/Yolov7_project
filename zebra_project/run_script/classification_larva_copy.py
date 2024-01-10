import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import pandas as pd

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1_dir_path', type=str, default=None)
    parser.add_argument('--img2_dir_path', type=str, default=None)
    parser.add_argument('--txt1_dir_path', type=str, default=None)
    parser.add_argument('--txt2_dir_path', type=str, default=None)
    # parser.add_argument('--output', type=str, default=None)
    
    return parser.parse_args()


argv = parse_arguments(sys.argv[1:])

img_list1 = os.listdir(argv.img1_dir_path)
txt_list1 = os.listdir(argv.txt1_dir_path)
img_list2 = os.listdir(argv.img2_dir_path)
txt_list2 = os.listdir(argv.txt2_dir_path)

img_list1.sort()
txt_list1.sort()
img_list2.sort()
txt_list2.sort()

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[3] - box1[1] + 1) * (box1[4] - box1[2] + 1)
    box2_area = (box2[3] - box2[1] + 1) * (box2[4] - box2[2] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

for i in range(len(img_list2)):

    image1 = Image.open(os.path.join(argv.img2_dir_path,img_list1[i]))
    imag1_size, imag2_size = image1.size
    txt_path1 = os.path.join(argv.txt1_dir_path,txt_list1[i])
    file1 = open(txt_path1, "r")
    strings = file1.read()
    file1.close()
    t1 = strings.split(' ')
    x = float(t1[1]) * float(imag1_size[0])
    y = float(t1[2]) * float(imag1_size[1]) 
    w = float(t1[3]) * float(imag1_size[0])
    h = float(t1[4]) * float(imag1_size[1])
    t1[1] = x-(w/2)
    t1[2] = y-(h/2)
    t1[3] = x+(w/2)
    t1[4] = y+(h/2)

    txt_path2 = os.path.join(argv.txt2_dir_path,txt_list2[i])
    file2 = open(txt_path2, "r")
    strings = file2.read()
    file2.close()
    t2 = strings.split(' ')
    x2 = float(t2[1]) * float(imag2_size[0])
    y2= float(t2[2]) * float(imag2_size[1]) 
    w2 = float(t2[3]) * float(imag2_size[0])
    h2 = float(t2[4]) * float(imag2_size[1])
    t2[1] = x2-(w2/2)
    t2[2] = y2-(h2/2)
    t2[3] = x2+(w2/2)
    t2[4] = y2+(h2/2)


    print('{0} IOU == {1}'.format(img_list1[i],IoU(t1,t2)))