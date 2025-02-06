import yaml

from functools import partial
import cv2
import torch
import numpy as np
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner.amp import autocast
from mmdet.datasets import CocoDataset

from paddleocr import draw_ocr

def read_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        
    return config

def generate_image_embeddings(prompt_image,
                              vision_encoder,
                              vision_processor,
                              projector,
                              device='cuda:0'
                              ):
    prompt_image = prompt_image.convert('RGB')
    inputs = vision_processor(images=[prompt_image],
                              return_tensors="pt",
                              padding=True)
    inputs = inputs.to(device)
    image_outputs = vision_encoder(**inputs)
    img_feats = image_outputs.image_embeds.view(1, -1)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    if projector is not None:
        img_feats = projector(img_feats)
    return img_feats

def adaptive_nms(bboxes, base_thr=0.5, dense_scene_thr=50):
    # Adjust IoU threshold based on object density
    if len(bboxes) > dense_scene_thr:
        nms_thr = base_thr - 0.2 # Lower threshold for dense scenes
    else:
        nms_thr = base_thr + 0.1 # Higher threshold for sparse scenes
    
    return nms_thr

def extract_object(target_image, pred_instances):
    detected_obj = []
    
    if pred_instances is None:
        print(F"Error: Prediction instances are missing")
        return detected_obj
    
    xyxy = pred_instances['bboxes']
    class_id = pred_instances['labels']
    confidence = pred_instances['scores']  
    
    if len(xyxy) == 0:
        print(f"Error: No objects were detected in the target image")
        return detected_obj

    index = len(confidence)
    for x in range(0, index):
        object = {}
        coord = xyxy[x]
        cropped_img = target_image.crop(tuple(coord))
        object['image'] = cropped_img
        object['xyxy'] = coord
        detected_obj.append(object)
    
    return detected_obj

def extract_texts(detected_objects, text_model, input_text):
    import re
    detected_obj_list = []
    
    if detected_objects is None or len(detected_objects) == 0:
        print(f"Error: No objects detected found for text extraction")
        return detected_obj_list

    if text_model is None:
        print(f"Error: Text model is not provided")
        return detected_obj_list
    
    if input_text is None or input_text.strip() == "":
        print(f"Error: Text prompt is empty. Please enter keywords")
        return detected_obj_list
    
    def check_text(extracted_text, input_text):
        input_text = re.findall(r'\b\w+\b', input_text)
        input_text = [x.lower() for x in input_text]
        
        image_text = []
        for item in extracted_text:
            words = re.findall(r'\b\w+\b', item)
            for word in words:
                word = word.lower()
                image_text.append(word)
                 
        num_of_matched_words = 0
        for word in image_text:
            if word in input_text:
                num_of_matched_words += 1
        return image_text, num_of_matched_words 

    
    for index, obj in enumerate(detected_objects):
        image = {}
        object_img = obj['image']
        object_coord = obj['xyxy']
        
        img = np.array(object_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        result = text_model.ocr(img)
        result = result[0]
        
        if result is None:
            print(f"Warning: OCR failed to extract text from object {index}")
            continue

        extracted_text = [line[1][0] for line in result]
        # extracted_score = [line[1][1] for line in result]
        
        if len(extracted_text) == 0:
            print(f"Warning: No text was extracted from object {index}")
            continue
        else:
            image_text, num_of_matched_words = check_text(extracted_text, input_text)
            
            if num_of_matched_words == 0:
                print(f"Warning: Input text and extracted text do not match from object {index}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image['image'] = Image.fromarray(img)
            image['text'] = image_text
            image['score'] = num_of_matched_words
            image['coord'] = object_coord
            
            detected_obj_list.append(image)
            
        if len(detected_obj_list) == 0:
            print(f"Warning: No matching text found in any detected objects.")
            return []
    return detected_obj_list
   
def extract_final_object(detected_object_list):
    
    if len(detected_object_list) == 0:
        print(f"Error: No object was detected. Unable to extract final object")
        return None
    
    from operator import itemgetter
    detected_object_list = sorted(detected_object_list, key=itemgetter('score'), reverse=True)
    
    return detected_object_list[0]

def run_image(runner,
              vision_encoder,
              vision_processor,
              padding_token,
              text_model,
              target_image,
              input_text,
              prompt_image,
            ):
    target_image = target_image.convert('RGB')
    add_padding = True
    if prompt_image is not None:
        texts = [['object'], ['']]
        projector = None
        
        if hasattr(runner.model, 'image_prompt_encoder'):
            projector = runner.model.image_prompt_encoder.projector
            
        prompt_embeddings = generate_image_embeddings(
            prompt_image,
            vision_encoder=vision_encoder,
            vision_processor=vision_processor,
            projector=projector)

        if add_padding == True:
            prompt_embeddings = torch.cat([prompt_embeddings, padding_token],
                                          dim=0)
            
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(
            p=2, dim=-1, keepdim=True)
        runner.model.num_test_classes = prompt_embeddings.shape[0]
        runner.model.setembeddings(prompt_embeddings[None])
    
    else:
        runner.model.setembeddings(None)
        texts = [[t.strip()] for t in input_text.split(',')]
        
    data_info = dict(img_id=0, img=np.array(target_image), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])    
        
    with autocast(enabled=False), torch.no_grad():
        if (prompt_image is not None) and ('texts' in data_batch['data_samples'][
                0]):
            del data_batch['data_samples'][0]['texts']
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        
    nms_thr = adaptive_nms(pred_instances.bboxes)
    score_thr = 0.3
    max_num_boxes = 100
    
    keep = nms(pred_instances.bboxes,
               pred_instances.scores,
               iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
        
    pred_instances = pred_instances.cpu().numpy()
    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    
    # img2 = np.array(target_image)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # img2 = BOUNDING_BOX_ANNOTATOR.annotate(img2, detections)
    # img2 = LABEL_ANNOTATOR.annotate(img2, detections, labels=labels)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # img2 = Image.fromarray(img2)
    # img2.show()
    
    detected_objects = extract_object(target_image, pred_instances)
    detected_objects_list = extract_texts(detected_objects, text_model, input_text)
    final_object = extract_final_object(detected_objects_list)
    
    # returning final image
    # image = final_object['image']
    # return image
    if final_object is None:
        print(f"Error: No object was detected. Unable to display final object")
        return None
    else:
        return final_object

def get_xyxy_from_yolo(label_file, img_w, img_h): 
    bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) < 5:
                continue  # Skip invalid lines
            
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
            
            x_center = x_center * img_w
            bbox_width = bbox_width * img_w
            y_center = y_center * img_h
            bbox_height = bbox_height * img_h  
            
            x1 = x_center - bbox_width / 2
            y1 = y_center - bbox_height / 2
            x2 = x_center + bbox_width / 2
            y2 = y_center + bbox_height / 2
            
            bboxes.append([x1, y1, x2, y2])  # Append each bounding box
    
    return bboxes  # Return a list of bounding boxes

def calculate_iou(pred_bbox, gt_bboxes):
    if not gt_bboxes:
        return 0  # No ground truth boxes
    
    best_iou = 0
    for gt_bbox in gt_bboxes:
        xA = max(pred_bbox[0], gt_bbox[0])
        yA = max(pred_bbox[1], gt_bbox[1])
        xB = min(pred_bbox[2], gt_bbox[2])
        yB = min(pred_bbox[3], gt_bbox[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        boxBArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

        iou = interArea / float(boxAArea + boxBArea - interArea) if boxAArea + boxBArea - interArea > 0 else 0
        best_iou = max(best_iou, iou)
    
    return best_iou


