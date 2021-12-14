from importlib import import_module
import os
from flask import Flask, render_template, Response
from utils.datasets import *
from utils.utils import *
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from base_camera import BaseCamera
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import yaml
import numpy as np
import queue
from threading import Thread


q = queue.Queue()

trash = []
checkTime = 0
im0 = None
start_record = False
fps = None
w = None
h = None
dataset = None
isChanged = 1
outputFileName = ""
vid_writer = None

startTime = time.time()
endTime = startTime + 15
tm = time.localtime(startTime)

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

app = Flask(__name__)

#video_source = 'rtsp://capstone:design@192.168.0.53/stream2'
video_source = 0

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = detect()


@app.route('/video_start')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@staticmethod
def set_video_source(source):
    Camera.video_source = source

def detect():
    global im0, start_record, fps, w, h, isChanged, dataset, startTime, endTime
    out, weights, imgsz = \
    'inference/output', 'weights/10_3.pt', 640
    source = '0'
    #source = 'rtsp://capstone:design@192.168.0.53/stream2'
    # device = torch_utils.select_device()
    
    # initialize deepsort
    deepsort = DeepSort("deep_sort_pytorch/deep/checkpoint/ckpt.t7",
        max_dist=0.2, min_confidence=0.3, max_iou_distance=0.9,
        max_age=14, n_init=1, nn_budget=30
        )

    # #Cam set
    # webcam = source == '0' or source.startswith(
    # 'rtsp') or source.startswith('http') or source.endswith('.txt')

    #webcam = 'rtsp://capstone:design@192.168.0.53/stream2'
    webcam = '0'
    
    
    device = "cpu"
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    model.to(device).float().eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    # half = False and device.type != 'cpu'
    half = True and device != 'cpu'

    if half:
        model.half()

    # Set Dataloader
    global vid_writer
    #dataset = LoadImages(source, img_size=imgsz)
    camera = cv2.VideoCapture(video_source)
    dataset = LoadStreams(source, img_size=imgsz)
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    global outputFileName
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device != 'cpu' else None  # run once
    
    before = []
    b_object = []
    object = []

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5,
            fast=True, classes=None, agnostic=False)
        t2 = torch_utils.time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s[i].copy()
            object = []
            person = []
            now = []

            if webcam:
                p, s, im0, path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        color = compute_color_for_id(id)
                        x1 = bboxes[0]
                        x2 = bboxes[2]
                        y1 = bboxes[1]
                        y2 = bboxes[3]
                        for i, box in enumerate(bboxes):
                            
                            # box text and bar
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                            cv2.rectangle(im0,(x1, y1),(x2,y2),color,3)
                            cv2.rectangle(im0,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                            cv2.putText(im0,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

                        # 모든 id와 좌표 저장
                        temp = [output[4], output[5], output[0], output[1], output[2]-output[0], output[3]-output[1]]

                        if(output[5]==0):
                            person.append(temp)
                        else:
                            object.append(temp)

                else:
                    deepsort.increment_ages()
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 6, im0.shape[1], im0.shape[0]

                
                # if isChanged == 1:
                #     isChanged = -1
                # else:
                #     isChanged = 1
                if outputFileName == "" or time.time() >= endTime or vid_writer == None:
                    print('start', startTime)
                    print('end', endTime)
                    print('name', outputFileName)
                    startTime = time.time()
                    endTime = startTime + 15
                    tm = time.localtime(startTime)
                    outputFileName = 'output_' + str(tm.tm_year) + str(tm.tm_mon) + str(tm.tm_mday) + str(tm.tm_hour) + str(tm.tm_min) + str(tm.tm_sec) + '.mov'
                    vid_writer = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

                print('%sDone. (%.3fs)' % (s, t2 - t1))
                return cv2.imencode('.jpg', im0)[1].tobytes()


def compute_color_for_id(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5001)
    with torch.no_grad():
        detect()

    