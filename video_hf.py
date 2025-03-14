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

def generate_image_embeddings2(prompt_images,
                              vision_encoder,
                              vision_processor,
                              projector,
                              device='cuda:0'
                              ):
    
    img_feats_list = []
    for img in prompt_images:
        img = img.convert('RGB')
        inputs = vision_processor(images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        image_outputs = vision_encoder(**inputs)
        img_feats = image_outputs.image_embeds.view(1, -1)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        
        if projector is not None:
            img_feats = projector(img_feats)
        
        img_feats_list.append(img_feats)
    
    return img_feats_list

def adaptive_nms(bboxes, base_thr=0.5, dense_scene_thr=50):
    # Adjust IoU threshold based on object density
    if len(bboxes) > dense_scene_thr:
        nms_thr = base_thr - 0.2 # Lower threshold for dense scenes
    else:
        nms_thr = base_thr + 0.1 # Higher threshold for sparse scenes
    
    return nms_thr

def extract_object(target_image, pred_instances):
    if pred_instances is None:
        raise gr.Error("Prediction instances are missing", duration=3)
    
    xyxy = pred_instances['bboxes']
    class_id = pred_instances['labels']
    confidence = pred_instances['scores']  
    
    if len(xyxy) == 0:
        raise gr.Error("No objects were detected in the target image", duration=3)
    
    detected_obj = []
    index = len(confidence)
    for x in range(0, index):
        object = {}
        coord = xyxy[x]
        cropped_img = target_image.crop(tuple(coord))
        # cropped_img.show()
        object['image'] = cropped_img
        object['xyxy'] = coord
        detected_obj.append(object)
    
    return detected_obj

def extract_texts(detected_objects, text_model, input_text):
    import re
    detected_obj_list = []
    
    if detected_objects is None or len(detected_objects) == 0:
        raise gr.Error("No objects detected found for text extraction", duration=3)

    if text_model is None:
        raise gr.Error("Text model is not provided")
    
    if input_text is None or input_text.strip() == "":
        raise gr.Error("Text prompt is empty. Please enter keywords", duration=3)
    
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
        common_words = []
        for word in image_text:
            if word in input_text:
                num_of_matched_words += 1
                common_words.append(word)
        return common_words, num_of_matched_words 

    for index, obj in enumerate(detected_objects):
        image = {}
        try:
            object_img = obj['image']
            object_coord = obj['xyxy']
            img = np.array(object_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            raise gr.Error(f"Error processing object {index}: {str(e)}", duration=3)
        
        result = text_model.ocr(img, cls = True)
        result = result[0]
        
        if result is None:
            # gr.Warning(f"OCR failed to extract text from object {index}", duration=3)
            print(f'Warning: OCR failed to extract text from object {index}')
            continue

        extracted_text = [line[1][0] for line in result]
        # extracted_score = [line[1][1] for line in result]
        
        if len(extracted_text) == 0:
            # gr.Warning(f"No text was extracted from object {index}", duration=3)
            print(f"No text was extracted from object {index}")
            continue
        else:
            image_text, num_of_matched_words = check_text(extracted_text, input_text)
            
            if num_of_matched_words == 0:
                # gr.Warning(f"Input text and extracted text do not match from object {index}", duration=3)
                print(f"Input text and extracted text do not match from object {index}")
                continue
            image['image'] = object_img
            image['text'] = image_text
            image['score'] = num_of_matched_words
            image['coord'] = object_coord
            
            detected_obj_list.append(image)

        if len(detected_obj_list) == 0:
            # gr.Warning("No matching text found in any detected objects.",duration=3)
            print('No matching text found in any detected objects.')
            return []
    return detected_obj_list
   
def extract_final_object(detected_object_list):
    
    if len(detected_object_list) == 0:
        print('Warning: No final object detected')
        return None

    from operator import itemgetter
    detected_object_list = sorted(detected_object_list, key=itemgetter('score'), reverse=True)
    
    return detected_object_list[0]

def run_image2(runner,
              vision_encoder,
              vision_processor,
              padding_token,
              text_model,
              target_image,
              input_text,
              prompt_images,
            ):
    target_image = target_image.convert('RGB')
    add_padding = True
    if prompt_images is not None:
        texts = [['object'], ['']]
        projector = None
        
        if hasattr(runner.model, 'image_prompt_encoder'):
            projector = runner.model.image_prompt_encoder.projector
        
        prompt_embeddings = generate_image_embeddings2(
            prompt_images,
            vision_encoder=vision_encoder,
            vision_processor=vision_processor,
            projector=projector)
        prompt_embeddings = torch.stack(prompt_embeddings).mean(dim=0)

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
        if (prompt_images is not None) and ('texts' in data_batch['data_samples'][
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
    if final_object is None:
        # gr.Warning("Error: No object was detected. Unable to display final object")
        return None
    else:
        # return final_object['image']
        return final_object  

def run_video(runner, vision_encoder, vision_processor, padding_token, text_model, target_video, input_text, prompt_images, frame_skip=10, output_video='output.mp4'):
    import os
    import subprocess
    cap = cv2.VideoCapture(target_video)
    if not cap.isOpened():
        raise gr.Warning("Error opening video file")
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue 

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        try:
            detected_object = run_image2(
                runner, vision_encoder, vision_processor, padding_token, text_model, frame_pil, input_text, prompt_images
            )
            if detected_object is None:
                print("Warning: No object detected in this frame.")
                out.write(frame)
                continue
        except gr.Error as e:
            print(f"Warning: {str(e)}")
            out.write(frame)
            continue
        
        if detected_object is not None:
            # Draw bounding box on frame
            bbox = detected_object['coord']
            # print(f'bbox: {bbox}')
            if bbox is not None:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        out.write(frame)
    
    cap.release()
    out.release()
    absolute_path = os.path.abspath(output_video)
    reencoded_video = absolute_path
    subprocess.run([
                    'ffmpeg', 
                    '-i', reencoded_video, # Input video file
                    '-c:v', 'h264', # Set video codec to H.264
                    '-c:a', 'aac', # Set audio codec to AAC
                    '-strict', 'experimental', # Allow experimental codecs
                    '-y', # Overwrite output file if it exists
                    reencoded_video # Output converted video file
                ], capture_output=True, text=True)
    return os.path.abspath(reencoded_video)