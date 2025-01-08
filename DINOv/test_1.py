import os
import cv2
import PIL.Image as Image
import helper_functions as hf
from demo import openset_task_1 as task
import torch

from dinov.BaseModel import BaseModel
from dinov import build_model
from utils.arguments import load_opt_from_config_file

REF_FOlDER = '/home/rse/prog/Dataset/Visual/VP001/ref_imgs/'
TEST_FOLDER = '/home/rse/prog/Dataset/Visual/VP001/'
CONF_FILE = '/home/rse/prog/Code/DINOv/configs/dinov_sam_coco_swinl_train.yaml'
CKPT = '/home/rse/prog/Code/DINOv/models/model_swinL.pth'

ref_image_file = REF_FOlDER + 'can_1.jpg'
target_image_file = TEST_FOLDER + '1.jpg'

generic_vp1 = {}
gray_img, img_cpy = hf.img_gray(ref_image_file)
mask = hf.binary_mask(gray_img)

img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)
img_cpy = Image.fromarray(img_cpy)

mask = Image.fromarray(mask).convert('RGB')

generic_vp1['image'] = img_cpy
generic_vp1['mask'] = mask

target_img = Image.open(target_image_file)

opt = load_opt_from_config_file(CONF_FILE)

model_sam = BaseModel(opt, build_model(opt)).from_pretrained(CKPT).eval().cuda()

@torch.no_grad()
def inference(generic_vp1, image2,*args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model=model_sam
        a= task.task_openset(model, generic_vp1, None, None, None,
                   None, None, None, None, image2, *args, **kwargs)
        img = Image.fromarray(a)
        img.save(f'/home/rse/prog/Code/DINOv/test_results/' + 'test_1.jpg')
        return a

inference(generic_vp1,target_img)




