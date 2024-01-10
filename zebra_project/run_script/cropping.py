import os
import platform
import shutil
import time
from pathlib import Path
from PIL import Image
import sys
import argparse


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    # parser.add_argument('--model', type=str, default=None)
    # parser.add_argument('--output', type=str, default=None)
    
    return parser.parse_args()


argv = parse_arguments(sys.argv[1:])
output_path = '/workspace/detect_infer/output'
output_ = os.path.join(output_path,argv.input)
output_list = os.listdir(os.path.join(output_,'img'))
for i in range(len(output_list)):

    img_path = os.path.join(output_,'img')
    txt_path = os.path.join(output_,'txt')
    img_list = os.listdir(img_path)
    txt_list = os.listdir(txt_path)

    img_list.sort()
    txt_list.sort()
    egg_img = []
    egg_txt = []
    larva_img = []
    larva_txt = []

    cropped_path = os.path.join('/workspace/cropped_egg',argv.input)
    if os.path.exists(cropped_path) == False:
        os.makedirs(cropped_path)
    fin_lar = os.path.join('/workspace/final_larva',argv.input)
    if os.path.exists(os.path.join(fin_lar,'img')) == False:
        os.makedirs(os.path.join(fin_lar,'img'))
    if os.path.exists(os.path.join(fin_lar,'txt')) == False:
        os.makedirs(os.path.join(fin_lar,'txt'))
    
    if len(img_list) != len(txt_list):
        img_li = img_list.copy()
        for v in img_li:
            im = v.split('.')[0]
            for j in txt_list:
                if im in j:
                    img_li.remove(v) 
        for r in img_li:
            larva_img.append(r)

    for b in range(len(txt_list)):
        file = open(os.path.join(txt_path,txt_list[b]), "r")
        strings = file.read()
        file.close()
        t = strings.split(' ')
        if int(t[0]) == 0:
            egg_txt.append(txt_list[b])
            for g in img_list:
                if txt_list[b].split('.')[0] in g:
                     egg_img.append(g)
        elif int(t[0]) == 1:
            larva_img.append(img_list[b])
            larva_txt.append(txt_list[b])

    egg_img.sort()
    egg_txt.sort()
    larva_img = set(larva_img)
    larva_img = list(larva_img)
    larva_img.sort()
    larva_txt.sort()

    for a in range(len(egg_img)):
    
        image1 = Image.open(os.path.join(img_path,egg_img[a]))
        imag1_size = image1.size
        file = open(os.path.join(txt_path,egg_txt[a]), "r")
        strings = file.read()
        file.close()
        t = strings.split(' ')
        x = float(t[1]) * float(imag1_size[0])
        y = float(t[2]) * float(imag1_size[1])
        w = float(t[3]) * float(imag1_size[0])
        h = float(t[4]) * float(imag1_size[1])

        croppedImage=image1.crop((x-(w/2),y-(h/2),x+(w/2),y+(h/2)))
        croppedImage.save(os.path.join('/workspace/cropped_egg', argv.input, (".".join(egg_img[a].split('.')[:-1])+'.jpg')))


    for c in range(len(larva_img)):
        shutil.move(os.path.join(img_path,larva_img[c]),os.path.join(fin_lar,'img'))
    for d in range(len(larva_txt)):
        shutil.move(os.path.join(txt_path,larva_txt[d]),os.path.join(fin_lar,'txt'))






