import yaml

from functools import partial
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner.amp import autocast
from mmdet.datasets import CocoDataset

from paddleocr import draw_ocr

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
MASK_ANNOTATOR = sv.MaskAnnotator()

def read_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        
    return config

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h

LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)

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
    xyxy = pred_instances['bboxes']
    class_id = pred_instances['labels']
    confidence = pred_instances['scores']
    
    score_thr = np.percentile(confidence, 70)
    
    index_list = []
    index = 0
    for x in confidence:
        if x >= score_thr:
            index_list.append(index)
        index += 1
    
    detected_obj = []
    for x in index_list:
        coord = xyxy[x]
        cropped_img = target_image.crop(tuple(coord))
        detected_obj.append(cropped_img)
        
    return detected_obj

def extract_texts(detected_objects, text_model, input_text):
    import re
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

    detected_obj_list = []
    for img in detected_objects:
        
        image = {}
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        result = text_model.ocr(img)
        result = result[0]
        extracted_text = [line[1][0] for line in result]
        extracted_score = [line[1][1] for line in result]
        
        image_text, num_of_matched_words = check_text(extracted_text, input_text)
        
        image['image'] = img
        image['text'] = image_text
        image['score'] = num_of_matched_words
        
        detected_obj_list.append(image)

        # boxes = [line[0] for line in result]
        # text = [texts.append(line[1][0]) for line in result]
        # score = [scores.append(line[1][1]) for line in result]
        
        # im_show = draw_ocr(img, boxes, texts, scores, font_path='PaddleOCR/doc/fonts/simfang.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.show()
    return detected_obj_list
   
def extract_final_object(detected_object_list):
    from operator import itemgetter
    
    detected_object_list = sorted(detected_object_list, key=itemgetter('score'), reverse=True)
    
    return detected_object_list[0]

# Starbucks, Dark Roast, Bold & Roasty, Premium Instant

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
    score_thr = 0.2
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
    detected_objects = extract_object(target_image, pred_instances)
    detected_objects_list = extract_texts(detected_objects, text_model, input_text)
    final_object = extract_final_object(detected_objects_list)
    
    # returning final image
    image = np.array(final_object['image'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image)
    return image


    
    
    
    
    
    