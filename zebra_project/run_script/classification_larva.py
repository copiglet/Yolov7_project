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
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--lar_out', type=str, default=None)
    # parser.add_argument('--output', type=str, default=None)
    
    return parser.parse_args()


argv = parse_arguments(sys.argv[1:])
csv = pd.read_csv(argv.csv,index_col=0)
lar_out = argv.lar_out
lar_dir = os.listdir(lar_out)[0]
lar_img = os.path.join(lar_out, lar_dir, 'img')
lar_txt = os.path.join(lar_out, lar_dir, 'txt')
img_list = os.listdir(lar_img)
txt_list = os.listdir(lar_txt)
img_list.sort()
txt_list.sort()
index_ls = csv.index
index_ls = set(index_ls)
index_ls = list(index_ls)
index_ls.sort()
case_dir = [index_ls[i] for i in range(len(index_ls)) if lar_dir[-1] in index_ls[i]]
if len(case_dir) == 0:
    print('Larva does not have previous cases')
    data = np.array(["None:0,0,0,0:0" for i in range(1800)])
    data = data.reshape(-1,data.shape[0])
    df = pd.DataFrame(data, index=[lar_dir], columns = csv.columns)
    csv = pd.concat([csv,df])
    for i in range(len(img_list)):

        image = Image.open(os.path.join(lar_img,img_list[i]))
        imag_size = image.size
        txt_path = os.path.join(lar_txt,txt_list[i])
        file1 = open(txt_path, "r")
        strings = file1.read()
        file1.close()
        t1 = strings.split(' ')
        x = float(t1[1]) * float(imag_size[0])
        y = float(t1[2]) * float(imag_size[1]) 
        w = float(t1[3]) * float(imag_size[0])
        h = float(t1[4]) * float(imag_size[1])
        t1[1] = x-(w/2)
        t1[2] = y-(h/2)
        t1[3] = x+(w/2)
        t1[4] = y+(h/2)
        t1_ls = [t1[1],t1[2],t1[3],t1[4]]
        csv.loc[lar_dir][img_list[i]] = f"larva:{t1_ls[0]},{t1_ls[1]},{t1_ls[2]},{t1_ls[3]}:1"
        # print(csv)
else:
    case_dir = set(case_dir)
    case_dir = list(case_dir)
    case_dir.sort()
    last_case = case_dir[-1]
    # print(last_case)
    data = np.array(["None:0,0,0,0:0" for i in range(1800)])
    data = data.reshape(-1,data.shape[0])
    df = pd.DataFrame(data, index=[lar_dir], columns = csv.columns)
    csv = pd.concat([csv,df])



    def IoU(box1, box2):
        # box = (x1, y1, x2, y2)
        # print(box1,box2)
        box1_area = (float(box1[2]) - float(box1[0]) + 1) * (float(box1[3]) - float(box1[1]) + 1)
        box2_area = (float(box2[2]) - float(box2[0]) + 1) * (float(box2[3]) - float(box2[1]) + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou

    for i in range(len(img_list)):

        image = Image.open(os.path.join(lar_img,img_list[i]))
        imag_size = image.size
        txt_path = os.path.join(lar_txt,txt_list[i])
        file1 = open(txt_path, "r")
        strings = file1.read()
        file1.close()
        t1 = strings.split(' ')
        x = float(t1[1]) * float(imag_size[0])
        y = float(t1[2]) * float(imag_size[1]) 
        w = float(t1[3]) * float(imag_size[0])
        h = float(t1[4]) * float(imag_size[1])
        t1[1] = x-(w/2)
        t1[2] = y-(h/2)
        t1[3] = x+(w/2)
        t1[4] = y+(h/2)
        t1_ls = [t1[1],t1[2],t1[3],t1[4]]
        if str(csv.loc[last_case][img_list[i]].split(":")[0]) == "egg":
            # org: csv.loc[lar_dir][img_list[i]] = f"larva:{t1_ls[0]},{t1_ls[1]},{t1_ls[2]},{t1_ls[3]}:1"
            # @TODO check this, Mulder was change this.
            csv.loc[last_case][img_list[i]] = f"larva:{t1_ls[0]},{t1_ls[1]},{t1_ls[2]},{t1_ls[3]}:1"
        t2 = str(csv.loc[last_case][img_list[i]]).split(":")[1].split(',')

        csv.loc[lar_dir][img_list[i]] = f"larva:{t1_ls[0]},{t1_ls[1]},{t1_ls[2]},{t1_ls[3]}:{IoU(t1_ls,t2)}"
        # print(csv)
        print('{0} IOU == {1}'.format(img_list[i],IoU(t1_ls,t2)))   


csv.to_csv(argv.csv)