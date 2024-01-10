from importlib.resources import path
import numpy as np
import json
from PIL import Image
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor, ToPILImage


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--eff_model', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    # parser.add_argument('--output', type=str, default=None)
    
    return parser.parse_args()


argv = parse_arguments(sys.argv[1:])
model_name = 'efficientnet-b0'  # b5

image_size = EfficientNet.get_image_size(model_name)
model = EfficientNet.from_name(model_name, num_classes=2)
# model = torch.load(argv.eff_model)
model._dropout = nn.Dropout(p=0.4, inplace=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu

model = model.to(device)
PATH = argv.model
model.load_state_dict(torch.load(PATH))
model.eval()

num_show_img = 4

class_names = {
    "0": "live",      # "0": "live"
    "1": "dead",   # "1": "dead"
}

csv = pd.read_csv(argv.csv,index_col=0)
img_ls = os.listdir(argv.input)
for img in img_ls:
    img_file = os.path.join(argv.input,img)
    img_RGB  = Image.open(img_file).convert('RGB')

    trans = transforms.Compose([transforms.Resize((224,224)), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # PIL to Tensor
    img_tensor = trans(img_RGB)
    img_tensor = img_tensor.to(device)
    img_RGB_tensor_from_PIL = img_tensor.unsqueeze(0)
    outputs = model(img_RGB_tensor_from_PIL)
    _, preds = torch.max(outputs, 1) 
    csv.loc[f"{argv.input.split('/')[-1]}"][img] = f"egg:{class_names[str(preds[0].cpu().numpy())]}"
    # print(csv)
    print('{0} is egg and {1}'.format(img,class_names[str(preds[0].cpu().numpy())]))


csv.to_csv(argv.csv)