import argparse
import os
import sys
from pathlib import Path
from app import auto_upload

import cv2
import torch
import time
import torch.backends.cudnn as cudnn

dumped = []
fixedTime = 0
isDumped = False;
normalShutdown = False;


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from deep_sort_pytorch.deep_sort import DeepSort
import datetime

def compute_color_for_id(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    deepsort = DeepSort("deep_sort_pytorch/deep/checkpoint/ckpt.t7",
                        max_dist=0.2, min_confidence=0.3, max_iou_distance=0.5
                        , max_age=70, n_init=3, nn_budget=100
                        )
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    previous = []
    preTrash = []
    trash = []
    global dumped
    global fixedTime
    global isDumped
    global normalShutdown

    save_vid = True,
    vid_path, vid_writer = None, None

    startTime = None
    endTime = None
    # tm = time.localtime(startTime)

    dt_now = datetime.datetime.now()

    save_path = 'output_' + dt_now.strftime('%Y%m%d%H%M%S') + '.mp4'
    upload_path = save_path

    # save_path = 'output_' + str(tm.tm_year) + str(tm.tm_mon) + str(tm.tm_mday) + str(tm.tm_hour) + str(
    #     tm.tm_min) + str(tm.tm_sec) + '.mp4'

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0

        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        # pred = non_max_suppression(pred, 0.4, 0.5,
        #                           fast=True, classes=None, agnostic=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()
        isRecord = False
        for i, det in enumerate(pred):  # per image
            p, s, im0 = path, '', im0s[i].copy()
            trash = []
            person = []
            current = []

            if webcam:
                p, s, im0, path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % im.shape[2:]

            # seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if det is not None and len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

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
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 3)
                            cv2.rectangle(im0, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                            cv2.putText(im0, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                        [255, 255, 255], 2)

                        coordinate = [output[4], output[5], output[0], output[1], output[2] - output[0],
                                output[3] - output[1]]

                        if (output[5] == 0):
                            person.append(coordinate)
                        else:
                            trash.append(coordinate)

                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                else:
                    deepsort.increment_ages()

                for p in person:
                    for o in trash:
                        if [p[0], o[0]] in current:
                            continue

                        if (p[2] + p[4]) >= o[2] and (p[3] <= (o[3] + o[5]) or p[3] + p[5] >= o[3]):
                            current.append([p[0], o[0]])

                        if (o[2] + o[4]) >= p[2] and (o[3] <= (p[3] + p[5]) or o[3] + o[5] >= p[3]):
                            current.append([p[0], o[0]])

                    # 위의 겹침 current 저장과 이전 겹침 previous 저장 비교하여 분리 확인 및 알림
                if len(previous) != 0:
                    for b in previous:
                        if b not in current:
                            print("투기 의심 상황 발생: " + str(b))
                            dumped.append(b[1])  # ID만 저장

                    # 새롭게 나타난 클래스(사람 제외) trash id 확인, 시간 10초 및 좌표 그대로 체크
                for o in trash:
                    if o in preTrash:
                        fixedTime = fixedTime + 1
                        print("Trash fixed time:" + str(fixedTime))

                preTrash = trash
                previous = current

                if fixedTime > 15:
                    isDumped = True;
                    print("isDumped: ", isDumped)
                    print("투기 상황 발생!");


            if save_vid:
                if vid_path != save_path:  # new video
                    print("new video")

                    vid_path = save_path

                    startTime = time.time()
                    endTime = startTime + 15


                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                        auto_upload(upload_path)
                        if isDumped:
                            with open("dumpedLog.txt", "a") as f:
                                f.write(upload_path + "\n")
                        isDumped = False;
                        normalShutdown = True;
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 12, im0.shape[1], im0.shape[0]

                    #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
                vid_writer.write(im0)

                if time.time() >= endTime:
                    # tm = time.localtime(startTime)
                    # save_path = 'output_' + str(tm.tm_year) + str(tm.tm_mon) + str(tm.tm_mday) + str(tm.tm_hour) + str(
                    #     tm.tm_min) + str(tm.tm_sec) + '.mp4'
                    dt_now = datetime.datetime.now()

                    upload_path = save_path;
                    save_path = 'output_' + dt_now.strftime('%Y%m%d%H%M%S') + '.mp4'
                    normalShutdown = False;

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # if len(det):
            #     # Rescale boxes from img_size to im0 size
            #     det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            #     # Print results
            #     for c in det[:, -1].unique():
            #         n = (det[:, -1] == c).sum()  # detections per class
            #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            #     # Write results
            #     for *xyxy, conf, cls in reversed(det):
            #         if save_txt:  # Write to file
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #             with open(txt_path + '.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #         if save_img or save_crop or view_img:  # Add bbox to image
            #             c = int(cls)  # integer class
            #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            #             annotator.box_label(xyxy, label, color=colors(c, True))
            #             if save_crop:
            #                 save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # # Stream results
            # im0 = annotator.result()
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

    # Print results

    if ~normalShutdown:
        vid_writer.release()
        auto_upload(save_path)
        if isDumped:
            with open("dumpedLog.txt", "a") as f:
                f.write(save_path + "\n")

    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)