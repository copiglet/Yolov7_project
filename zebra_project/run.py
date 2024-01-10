import argparse
from operator import index
import os
import platform
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np
import sys
import pandas as pd
import glob

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    # parser.add_argument('--txt_dir_path', type=str, default=None)
    # parser.add_argument('--output', type=str, default=None)
    
    return parser.parse_args()


argv = parse_arguments(sys.argv[1:])
detect_infer = '/workspace/detect_infer/output'
cls_model = '/workspace/models/egg_model.pt'
det_model = '/workspace/models/best_yolov4-csp-results.pt'
eff_model = '/workspace/models/efficientnet-b0-355c32eb.pth'
remove_input = '/workspace/input'

lar_out = '/workspace/final_larva'
input_list = os.listdir(argv.input)
input = os.path.join(argv.input,input_list[0])
input_array = np.array(input_list)
csv_file = '/workspace/output.csv'

egg_list = []
larva_list = []
# detection_part
os.system('python3 /workspace/detection.py --weights ' + det_model + ' --img 416 --conf 0.4 --source ' + input + ' --save-txt')
os.system('python3 /workspace/run_script/cropping.py --input ' + input.split('/')[-1])


# classification_part(larva)

os.system('python3 /workspace/run_script/classification_larva.py --csv ' + csv_file + ' --lar_out ' + lar_out)


# classification_part
input_dir = input.split('/')[-1]
crop_path = os.path.join('/workspace/cropped_egg',input_dir)
os.system('python3 /workspace/run_script/classification_egg.py --input ' + crop_path + ' --csv ' + csv_file + ' --eff_model ' + eff_model +' --model ' + cls_model)


shutil.rmtree(crop_path)
shutil.rmtree(os.path.join(detect_infer,os.listdir(detect_infer)[0]))
shutil.rmtree(os.path.join(lar_out,os.listdir(lar_out)[0]))
shutil.rmtree(os.path.join(remove_input,os.listdir(remove_input)[0]))




