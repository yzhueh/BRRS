import os
import sys
from pathlib import Path
import cv2
from PIL import Image, ImageDraw
import torch
import math
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
from utils.datasets import letterbox
from utils.plots import Colors
import numpy as np
import pdb


class Detector(object):
    def __init__(self, weight_path="weights/best.pt",
                 device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.2,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 input_size=(416, 416)  # resize of input image.
                 ):
        super(Detector, self).__init__()
        self.weight_path = weight_path
        assert self.weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.input_size = input_size
        # Load model
        self.device = select_device(device)
        self.model = attempt_load(self.weight_path, map_location=self.device)  # load FP32 model
        self.model.eval()

    def get_result(self, img):
        """

        :param img:  ndarray, shape is 'h,w,c'
        :return:     list , [ {'position':ndarray:[x1,y1,x2,y2,p],'label':str} ,
                              {...},
                              ....
                            ]
                    or None.
        """
        pointsList = []
        im0 = img
        img = letterbox(img, self.input_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.Tensor(img.copy())
        # Inference
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.float().to(self.device)
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        rs = det[:,0:4]
        # print(rs)
        return rs

if __name__ == '__main__':
    path = r'E:\code\data\images\1.jpg'
    model = Detector()
    img = cv2.imread(path)
    results = model.get_result(img)
    count = 1
    for result in results:
        print(result)
        cut_img = img[int(result[1]):int(result[3]),int(result[0]):int(result[2])]
        cv2.imwrite('cut_imgs'+'\\'+str(count)+'.jpg',cut_img)
        count += 1






