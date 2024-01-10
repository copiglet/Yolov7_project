import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import glob
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
import shutil
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, non_max_suppression_custom, non_max_suppression_custom2, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import pmxevent as pevent 
import pandas as pd
import json


weights, view_img, save_txt, imgsz, trace = './models/best.pt', False, True, 640, True
save_img = os.environ.get('SAVE_IMG', "false").lower() == "true"
debug_print = os.environ.get('DEBUG_PRINT', "false").lower() == "true"
image_path = os.environ.get('IMAGE_PATH', './inference/images')
web_debug = os.environ.get('WEB_DEBUG', "false").lower() == "true"
webcam = False

print('==========CONFIG===========')
print(f" - weights:{weights}")
print(f" - view_img:{view_img}")
print(f" - save_txt:{save_txt}")
print(f" - imgsz:{imgsz}")
print(f" - trace:{trace}")
print(f" - save_img:{save_img}")
print(f" - debug_print:{debug_print}")
print(f" - image_path:{image_path}")
print(f" - web_debug:{web_debug}")
print("==========================")

# Initialize
if debug_print:
    set_logging()
else:
    set_logging(50)

device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA


# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, 640)
    
if half:
    model.half()
    
# Second-stage classifier
classify = False
if classify:
    modelc = load_classifier(name='resnet101', n=2)  # initialize
    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
# colors = [80,205,180,120]

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1



# return code, detail
#
def run_detect(input_id, pt_path = '',btr_type = '', dir_num = '', df = pd.DataFrame([""])):
    global old_img_w, old_img_h, old_img_b 
    image_pt_path = image_path if pt_path == '' else pt_path
    ret_json = {"id": input_id, "detail": []}
    input_path = os.path.join(image_pt_path, input_id)
    print('run_detect:' + input_path)
    if not os.path.isdir(input_path):
        print(404, {"error": f"input path fail: {input_path}"})
        return 404, {"error": f"input path fail: {input_path}"} 

    im_path = input_path
    save_dir = os.path.join('/home/huvio/Project/huvio/data/ai_data/2023-03-15', btr_type, dir_num, im_path.split('/')[-1])  # increment run
    save_dir = Path(save_dir, exist_ok=True)
    # save_dir.mkdir(parents=True, exist_ok=True)
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    dataset = LoadImages(im_path, img_size=imgsz, stride=stride)
    emp = []
    for j in range(21):
        emp.append('NaN')


    t0 = time.time()
    # event fire
    pevent.enterjob(im_path)
    for path, img, im0s, vid_cap in dataset:
        # detail_js = {"fn": path.replace(image_pt_path, ""), "cls": -1, "score": 0,"boxs": []}
        pth = path.split('/ag_data')[-1]

        detail_js = {"fn": pth, "cls": -1, "boxs": []}

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=True)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]
        t2 = time_synchronized()
        # pred는 모델로부터 얻은 출력 텐서입니다.
        # pred = non_max_suppression_custom(pred, conf_thres_dict={0: float(thres.split(',')[0]), 1: float(thres.split(',')[1]), 2: float(thres.split(',')[2]), 3: float(thres.split(',')[3]), 4: float(thres.split(',')[4]), 5: float(thres.split(',')[5]), 6: float(thres.split(',')[6]), 7: float(thres.split(',')[7]), 8: float(thres.split(',')[8]), 9: float(thres.split(',')[9])}, iou_thres=0.1)
        pred, top_score_boxes = non_max_suppression_custom2(pred, 0.1, 0.1, classes=[0,1,2,3,4,5,6,7,8,9], agnostic=True)
        # Apply NMS
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # print("im0.shape:", im0.shape)
            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                score = 0.0
                class0 = -1
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh


                    # add result
                    # detail_js["cls"] = int(cls)
                    zh, zw, zc = im0.shape
                    xywh_world = [round(xywh[0]*zw), round(xywh[1]*zh), round(xywh[2]*zw), round(xywh[3]*zh)]
                    
                    detail_js["boxs"].append({"cls": int(cls), "conf": round(float(conf), 3), "xywh": xywh_world})
                    if score < conf:
                        score = conf
                        class0 = int(cls)
                detail_js["cls"] = class0
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            detail_js["tm"] =  (1E3 * (t2 - t1)) + (1E3 * (t3 - t2))

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # cv2.resize(im0,(640,640))
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        # end of one img
        if len(detail_js["boxs"]):
            box_clss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 10 class
            for box in detail_js["boxs"]:
                box_clss[box["cls"]] += 1
            max_value = max(box_clss)
            max_index = box_clss.index(max_value)
            detail_js["cls"] = int(max_index)
            if int(max_index) == 0:
                class1 = 'pt1_g'
  
            elif int(max_index) == 1:
                class1 = 'pt1_p'
 
            elif int(max_index) == 2:
                class1 = 'pt1_d'

            elif int(max_index) == 3:
                class1 = 'pt1_t'
 
            elif int(max_index) == 4:
                class1 = 'pt1_f'

            elif int(max_index) == 5:
                class1 = 'pt2_g'
 
            elif int(max_index) == 6:
                class1 = 'pt2_p'
  
            elif int(max_index) == 7:
                class1 = 'pt2_d'

            elif int(max_index) == 8:
                class1 = 'pt2_t'
    
            elif int(max_index) == 9:
                class1 = 'pt2_f'
      

        if debug_print:
            print("detail_js:", detail_js)
        ret_json["detail"].append(detail_js)


    print(f'Done. ({time.time() - t0:.3f}s)')
    # event fire
    pevent.finishjob('./'+str(save_dir))

    return 200, ret_json, emp



def get_png_sub_dirs(rootdir, out_list):
    for file1 in os.listdir(rootdir):
        d = os.path.join(rootdir, file1)
        if os.path.isdir(d):
            pngcnt = 0
            for ff in os.listdir(d):
                f = os.path.join(d, ff)
                if f.endswith(".png"):
                    pngcnt += 1
            if pngcnt > 9:
                out_list.append((rootdir, file1, d))
            get_png_sub_dirs(d, out_list)

# ================================================
PATH_WEBD="./static/webd"
def cleanWebDebug():
    if not os.path.exists(PATH_WEBD):
        os.makedir(PATH_WEBD)
    os.system("rm -f " + PATH_WEBD + "/*")

def writeWebDebug(fpath, jso):
    with open(os.path.join(PATH_WEBD, fpath), "w") as outfile:
        json.dump(jso, outfile)
# ================================================



# run batch prediction
# btr_type = m or p
# ymd = 2023-03-15
def ng_batch_detect(rootdir, btr_type, ymd):
    dir_num = rootdir.split('/')[-1]
    json_dir = os.path.join(f'/home/huvio/Project/huvio/data/ai_data/{ymd}', btr_type, dir_num, 'output')

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    dstdir = json_dir

    # fn_res = os.path.join(dstdir, "result.csv") 
    hour_n = rootdir.split('/')[-1]
    batch_json = {"hour": hour_n, "data": []}
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)

    out_list = [] # clear
    get_png_sub_dirs(rootdir, out_list)
    print('batch enter:', rootdir, " total:", len(out_list))
    print("====")

    # init and set
    pred_list = []
    for i, item in enumerate(out_list):
        pred_list.append({"no": i, "item": item})
    if web_debug:
        cleanWebDebug()
        writeWebDebug('pred_list.json', pred_list)

    dorun = True
    out_lines = []
    for jobitem in pred_list:
        item = jobitem["item"]
        pt_path, req_id, f_dir = item

        if dorun:
            code, res_js, emp = run_detect(req_id, pt_path, btr_type, dir_num)
            if web_debug:
                writeWebDebug(f'{jobitem["no"]:05}_item.json', {"no": jobitem["no"], "result":
                    res_js})
            # txtcode = ["", "GOOD", "ERROR", "TAB"] 
            if code != 200:
                print('batch run:', f_dir, ':ERROR')
            else:
                if debug_print:
                    print('batch run:', f_dir, ':', "None")
            batch_json['data'].append(res_js)
            
    json_string = json.dumps(batch_json, indent=4)
    with open(os.path.join(dstdir,'result.json'), 'w') as f:       
        f.write(json_string)
    
    
    return {"dstdir": dstdir, "total": len(out_list), "result_file": os.path.join(dstdir,'result.json')}


#
# run :   DEBUG_PRINT=false python detect_web.py 
#
if __name__ == '__main__':
    print('AI TOOL RUN')

    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--testdir", type=str,
                        help="test path", default="")
    parser.add_argument("-btrtype", "--btrtype", type=str,
                        help="plus(p) or minus(m)", default="")
    parser.add_argument("-ymd", "--ymd", type=str,
                        help="yyyy-mm-dd 2023-03-15", default="")
    args = parser.parse_args()

    print(args);

    if os.path.exists(args.testdir) and len(args.btrtype)==1 and len(args.ymd)==10:
        ng_batch_detect(args.testdir, args.btrtype, args.ymd)
    else:
        print(f"path not found : {args.testdir}")


