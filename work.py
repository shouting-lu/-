import os
import cv2
import sys
import time
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import list_dataset, cv2_transform
from datasets.ava_dataset import Ava 
from datasets.ava_eval_helper import read_labelmap 
from datasets.meters import AVAMeter
from core.optimization import *
#from cfg import parser
from cfg.parser import load_config
from core.utils import *
from core.region_loss import RegionLoss, RegionLoss_Ava
from core.model import YOWO, get_fine_tuning_parameters


def main(cfg, opt):
    ROI_ACTONS = [
        {'id': 8, 'name': 'lie/sleep'},
        {'id': 14, 'name': 'walk'},
        {'id': 15, 'name': 'answer phone'},
        {'id': 37, 'name': 'listen (e.g., to music)'},
        {'id': 54, 'name': 'smoke'},
        {'id': 57, 'name': 'text on/look at a cellphone'},
        {'id': 62, 'name': 'work on a computer'},
        {'id': 63, 'name': 'write'}, 
        {'id': 79, 'name': 'talk to (e.g., self, a person, a group)'},
    ]
    ROI_ACTIONS_ID = [8, 14, 15, 37, 54, 57, 62, 63, 79]

    ####### Create model
    # ---------------------------------------------------------------
    model = YOWO(cfg)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None) # in multi-gpu case

    ####### Inference parameters
    # ---------------------------------------------------------------
    labelmap, _       = read_labelmap(cfg.DEMO.LABEL_FILE_PATH)
    num_classes       = cfg.MODEL.NUM_CLASSES

    anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors       = cfg.SOLVER.NUM_ANCHORS

    clip_length		  = opt.clip_length
    crop_size 		  = opt.test_crop_size # default 224, increase the number for better detection performance
    nms_thresh        = opt.nms_thresh  # default 0.5
    conf_thresh_valid = opt.conf_thresh_valid  # default 0.5 ,For more stable results, this threshold is increased!

    ####### Prepare model
    # ---------------------------------------------------------------
    print("===================================================================")
    print('loading checkpoint {}'.format(opt.pretrained))
    checkpoint = torch.load(opt.pretrained)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded model score: ", checkpoint['score'])
    print("===================================================================")
    del checkpoint

    model.eval()


    ####### Data preparation and inference
    cap = cv2.VideoCapture(opt.input_video)
    frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    origin_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    origin_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cnt = 1
    queue = []
    vid_writer = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if len(queue) <= 0: # At initialization, populate queue with initial frame
            for i in range(clip_length):
                queue.append(frame)

        # Add the read frame to last and pop out the oldest one
        queue.append(frame)
        queue.pop(0)

        cnt += 1
        print("frame: ", cnt)

        #frame = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        #import ipdb; ipdb.set_trace()
        # Resize images
        #imgs = [cv2_transform.resize(crop_size, img) for img in imgs]
        imgs = [cv2_transform.resize(crop_size, img) for img in queue]

        #frame = imgs[-1]
        #frame = img = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(cfg.DATA.MEAN, dtype=np.float32),
                np.array(cfg.DATA.STD, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        imgs = torch.unsqueeze(imgs, 0)


        # Model inference
        with torch.no_grad():
            output = model(imgs)

            preds = []
            all_boxes = get_region_boxes_ava(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)

            #import ipdb; ipdb.set_trace()

            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                
                for box in boxes:
                    x1 = float(box[0]-box[2]/2.0)
                    y1 = float(box[1]-box[3]/2.0)
                    x2 = float(box[0]+box[2]/2.0)
                    y2 = float(box[1]+box[3]/2.0)
                    det_conf = float(box[4])
                    cls_out = [det_conf * x.cpu().numpy() for x in box[5]]
                    preds.append([[x1,y1,x2,y2], cls_out])

        #import ipdb; ipdb.set_trace()
        for dets in preds:
            x1 = int(dets[0][0] * origin_w)
            y1 = int(dets[0][1] * origin_h)
            x2 = int(dets[0][2] * origin_w)
            y2 = int(dets[0][3] * origin_h)

            cls_scores = np.array(dets[1])
            rec=[8,11,12,14,15,57,62,74,79]
            remo=[]

            indices = np.argsort(cls_scores)[-5:]
            for i,cls_ind in enumerate(indices):
                if cls_ind not in rec:
                    remo.append(cls_ind)
            for i,cls_ind in enumerate(remo):
                indices.delete(cls_ind)
            scores = cls_scores[indices]
            indices = list(indices)
            scores = list(scores)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            if len(scores) > 0:
                blk   = np.zeros(frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                #import ipdb; ipdb.set_trace()
                for i, cls_ind in enumerate(indices):
                    if opt.use_roi and cls_ind not in ROI_ACTIONS_ID:
                        continue

                    text.append(str(labelmap[cls_ind]['name']))

                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*i))

                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                
                frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(frame, text[t], coord[t], font, 0.5, (255, 0, 0), 1)

                state=''
                if 12 in indices or 14 in indices:
                    state = 'Off Duty'
                    cv2.rectangle(frame, (x1+2,y2-18), (x1+2+text_size[-1][0],y2+text_size[-1][1]-22), (0,255,0), 2)
                    cv2.putText(frame, state, (), font, 0.5, (255, 0, 0), 1)
                elif 11 in indices or 62 in indices:
                    state = 'On Duty'
                    cv2.rectangle(frame, (x1+2,y2-18), (x1+2+text_size[-1][0],y2+text_size[-1][1]-22), (0,255,0), 2)
                    cv2.putText(frame, state,(x1+3,y2-12) , font, 0.5, (255, 0, 0), 1)
                    
        if vid_writer is None:
            save_path = os.path.join(opt.save_path, 'output.mp4')
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (origin_w, origin_h))
        else:
            vid_writer.write(frame)

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        
        frames_path = os.path.join(opt.save_path, 'frames')
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        cv2.imwrite(frames_path+'/{:05d}.jpg'.format(cnt), frame) # save figures if necessay

    cap.release()
    vid_writer.release()

if __name__ == "__main__":
    ####### Load configuration arguments
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Provide working status supervision.")
    parser.add_argument("--input-video",help="Path to the video file",default="demo.mp4",type=str)
    parser.add_argument("--test-crop-size",help="Image crop size for inference",default=224,type=int)
    parser.add_argument("--clip-length",help="Video clip length for inference",default=16,type=int)
    parser.add_argument("--pretrained",help="Pretrained model weights",default="pretrained/work_16f_s2_best_ap_01719.pth",type=str)
    parser.add_argument('--save-video', action='store_true', help='save inferenced videos')
    parser.add_argument("--save-path",help="Path to save the video file",default="inference",type=str)
    parser.add_argument('--use-roi', action='store_true', help='save inferenced videos')
    parser.add_argument("--action-list",help="Path to the action list file",default="cfg/action_list_v2.pbtxt",type=str)
    parser.add_argument("--conf-thresh-valid",help="conf-thresh-valid",default=0.5,type=float)
    parser.add_argument("--nms-thresh",help="nms_thresh",default=0.5,type=float)
    parser.add_argument("--cfg",dest="cfg_file",help="Path to the config file",default="cfg/work.yaml",type=str)
    parser.add_argument("opts",help="See cfg/defaults.py for all options",default=None,nargs=argparse.REMAINDER,)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()
    cfg = load_config(args)

    main(cfg, args)


