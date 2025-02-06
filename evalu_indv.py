import os
from PIL import Image, ImageDraw
import helper_functions as hf
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import pandas as pd

from functools import partial 
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmengine.dataset import Compose
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)

CSV_FILE = 'Dataset/Visual/VP004/test_a.csv'

def resize_image(image_path, target_size=(640, 640)):
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    target_aspect = target_size[0] / target_size[1]
    img_aspect = img_width / img_height
    
    if img_aspect > target_aspect:
            # Image is wider than target: scale by width
            new_width = target_size[0]
            new_height = int(target_size[0] / img_aspect)
    else:
        # Image is taller than target: scale by height
        new_height = target_size[1]
        new_width = int(target_size[1] * img_aspect)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    new_img = Image.new("RGB", target_size, (255, 255, 255)) # White background
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(resized_img, (paste_x, paste_y))
    
    return new_img

# def get_xyxy_from_yolo(label_file, img_w, img_h, class_name='object'):
#     bboxes = []
#     with open(label_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if not parts:
#                 continue
            
#             x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
            
#             x_center *= img_w
#             bbox_width *= img_w
#             y_center *= img_h
#             bbox_height *= img_h
            
#             x1 = x_center - bbox_width / 2
#             y1 = y_center - bbox_height / 2
#             x2 = x_center + bbox_width / 2
#             y2 = y_center + bbox_height / 2
            
#             bboxes.append([x1, y1, x2, y2])
            
#     return bboxes

def get_xyxy_from_yolo(label_file, img_w, img_h, class_name='object'): 
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if not parts:
                continue
            
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
            
            x_center = x_center * img_w
            bbox_width = bbox_width * img_w
            y_center = y_center * img_h
            bbox_height = bbox_height * img_h  
            
            x1 = x_center - bbox_width / 2
            y1 = y_center - bbox_height / 2
            x2 = x_center + bbox_width / 2
            y2 = y_center + bbox_height / 2
            
            yolo_bbox = [x1, y1, x2, y2]
            
            return yolo_bbox

# def get_highest_conf_bbox(detections):
#     if detections is None or 'bboxes' not in detections or 'scores' not in detections:
#         print('Prediction instances are missing or incomplete')
#         return None

#     xyxy = detections['bboxes']
#     confidence = detections['scores']
    
#     if len(xyxy) == 0:
#         print('No objects were detected in the target image')
#         return None

#     max_conf_index = np.argmax(confidence)
#     highest_conf_bbox = xyxy[max_conf_index]
#     highest_conf_score = confidence[max_conf_index]
    
#     return highest_conf_bbox, highest_conf_score

def get_highest_conf_bbox(detections):
    if detections is None:
        print('Prediction instances are missing')
        return None

    xyxy = detections['bboxes']
    confidence = detections['scores']
    
    if len(xyxy) == 0:
        print('No objects were detected in the target image')
        return None

    max_conf_index = np.argmax(confidence)
    highest_conf_bbox = xyxy[max_conf_index]
    highest_conf_score = confidence[max_conf_index]
    
    return highest_conf_bbox

# def calculate_iou(pred_bboxes, gt_bboxes):
#     iou_scores = []

#     for pred_bbox in pred_bboxes:
#         best_iou = 0
#         for gt_bbox in gt_bboxes:
#             pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
#             gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
            
#             pred_area = max(0, pred_x2 - pred_x1) * max(0, pred_y2 - pred_y1)
#             gt_area = max(0, gt_x2 - gt_x1) * max(0, gt_y2 - gt_y1)
            
#             if pred_area <= 0 or gt_area <= 0:
#                 continue
            
#             x1 = max(pred_x1, gt_x1)
#             y1 = max(pred_y1, gt_y1)
#             x2 = min(pred_x2, gt_x2)
#             y2 = min(pred_y2, gt_y2)
            
#             intersection_width = max(0, x2 - x1)
#             intersection_height = max(0, y2 - y1)
#             intersection_area = intersection_width * intersection_height

#             iou = intersection_area / (pred_area + gt_area - intersection_area)
            
#             best_iou = max(best_iou, iou)

#         iou_scores.append(best_iou)
    
#     return iou_scores

def calculate_iou(pred_bbox, gt_bbox):
    gt_x1 = gt_bbox[0]
    gt_y1 = gt_bbox[1]
    gt_x2 = gt_bbox[2]
    gt_y2 = gt_bbox[3]
    
    pred_x1 = pred_bbox[0]
    pred_y1 = pred_bbox[1]
    pred_x2 = pred_bbox[2]
    pred_y2 = pred_bbox[3]
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    
    if pred_area <= 0 or gt_area <= 0:  # Check for invalid bounding box areas
        return 0  # Return 0 IOU for invalid bboxes
    
    x1 = max(pred_x1, gt_x1)
    y1 = max(pred_y1, gt_y1)
    x2 = min(pred_x2, gt_x2)
    y2 = min(pred_y2, gt_y2)
    
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    iou = intersection_area / (pred_area + gt_area - intersection_area)

    return iou


def calculate_accuracy(iou_scores, iou_threshold = 0.5):    
    if not iou_scores:
        print(f"iou score is empty")
        return 0.0
        
    iou_scores = np.array(iou_scores)
    correct_detections = np.sum(iou_scores >= iou_threshold)
    total_detections = len(iou_scores)
    
    if total_detections > 0:
        accuracy = correct_detections / total_detections
    else:
        accuracy = 0.0
        
    return accuracy

if __name__ == '__main__':

    paddle_ocr = PaddleOCR(lang='en')
    demo_cfg = hf.read_config("config.yaml")
    visual_cfg = Config.fromfile(demo_cfg['Visual']['config_path'])
    
    visual_cfg.load_from = demo_cfg['Visual']['model_path']
    visual_cfg.work_dir = demo_cfg['Visual']['work_dir']
    
    
    if 'runner_type' not in visual_cfg:
        runner = Runner.from_cfg(visual_cfg)
    else:
        runner = RUNNERS.build(visual_cfg)
        
    runner.call_hook('before_run')
    runner.load_or_resume()
    pipepline = visual_cfg.test_dataloader.dataset.pipeline
    pipepline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipepline)
    runner.model.eval()
    
    clip_model = "openai/clip-vit-base-patch32"
    vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
    processor = AutoProcessor.from_pretrained(clip_model)
    device = 'cuda:0'
    vision_model.to(device)
    
    texts = [' ']
    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    text_model = CLIPTextModelWithProjection.from_pretrained(clip_model)
    text_model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = text_model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])
    txt_feats = txt_feats[0].unsqueeze(0)
    
    df = pd.read_csv(CSV_FILE)
    iou_scores = []
    for index, row in df.iterrows():
        print(f"Target Image: {row['target_img']}")
        print(f"Ref image: {row['ref_img']}")
        print(f"Annotated image: {row['annotated_img']}")
        print(f"Label file: {row['label_file']}")
        
        target_img = row['target_img']
        target_img = resize_image(target_img)
        ref_img = Image.open(row['ref_img'])
        input_text = None
        
        annotated_img = Image.open(row['annotated_img'])
        a_width, a_height = annotated_img.size
        label_file = row['label_file']
        
        yolo_bbox = get_xyxy_from_yolo(label_file, a_width, a_height)
        detections = hf.run_image_evaluation(runner, vision_model, processor, txt_feats, 
                     paddle_ocr, target_img, input_text, ref_img)
        
        # if detections is None or 'bboxes' not in detections or 'scores' not in detections:
        #     print("Error: Detections are missing required keys.")
        #     iou_scores.append(0.0)
        #     continue
        highest_conf_bbox = get_highest_conf_bbox(detections)

        # result = get_highest_conf_bbox(detections)
        # if result is None:
        #     print("Error: No detections found.")
        #     iou_scores.append(0.0)
        #     continue

        # highest_conf_bbox, highest_conf_score = result
        
        if yolo_bbox is None or highest_conf_bbox is None:
            print("Error: pred_bbox or gt_bbox is None. Skipping IOU calculation")
            iou_score = 0.0
            iou_scores.append(iou_score)
            continue

        # if highest_conf_bbox is None or not yolo_bboxes:
        #     print("Error: pred_bbox or gt_bbox is None. Skipping IOU calculation")
        #     iou_scores.append(0.0)
        #     continue

        # iou_scores_list = calculate_iou([highest_conf_bbox], yolo_bboxes)
        # iou_score = max(iou_scores_list)  # Take the highest IoU for evaluation

        # if iou_score is not None:
        #     print(f"IOU score: {iou_score}")
        #     iou_scores.append(iou_score)
        # else:
        #     print("Error: IOU score is None. Skipping this entry")
        
        iou_score = calculate_iou(highest_conf_bbox, yolo_bbox)
        if iou_score is not None:
            print(f"IOU score: {iou_score}")
            iou_scores.append(iou_score)
        else:
            print("Error: IOU score is None. Skipping this entry")


        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
    accuracy = calculate_accuracy(iou_scores, iou_threshold=0.5)
    print(f"Accuracy: {accuracy}")
