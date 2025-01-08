import os
import cv2
import PIL.Image as Image
import numpy as np


def img_gray(file):
    image = cv2.imread(file)
    image = cv2.resize(image, [640, 480])
    
    image_copy = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    return gray, image_copy

    
def binary_mask(gray_image):
    _, mask = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)
    
    return mask

def draw_bbox_with_contours(gray_img, thresh):
    
    canny_output = cv2.Canny(gray_img, threshold1=thresh, threshold2=thresh*2)
    
    contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        all_contours = np.vstack(contours)
        
        hull = cv2.convexHull(all_contours)
        
        x, y, w, h = cv2.boundingRect(hull)
        print(f'x: {x}, y: {y}, w: {w}, h: {h}')
        return x, y, w, h
    else:
        print('No possible object found')
        return None
    
def show_mask_opencv(mask):
    h, w = mask.shape[-2:]
    mask_image = np.zeros((h, w, 3), dtype=np.uint8)

    mask_image[mask > 0] = [255, 255, 255] 

    return mask_image
    

