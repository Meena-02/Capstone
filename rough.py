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

# highest_conf_bbox, highest_conf_score = get_highest_conf_bbox(detections)


# if yolo_bbox is None or highest_conf_bbox is None:
#     print("Error: pred_bbox or gt_bbox is None. Skipping IOU calculation")
#     iou_score = 0.0
#     iou_scores.append(iou_score)
#     continue

# iou_score = calculate_iou(highest_conf_bbox, yolo_bbox)
# if iou_score is not None:
#     print(f"IOU score: {iou_score}")
#     iou_scores.append(iou_score)
# else:
#     print("Error: IOU score is None. Skipping this entry")
        