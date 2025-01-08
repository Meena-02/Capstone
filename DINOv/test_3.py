import os
import cv2
import PIL.Image as Image
import helper_functions as hf
from demo import openset_task_3 as task
import torch
import numpy as np

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from dinov.BaseModel import BaseModel
from dinov import build_model
from utils.arguments import load_opt_from_config_file

REF_FOlDER = '/home/rse/prog/Dataset/Visual/VP001/ref_imgs/'
TEST_FOLDER = '/home/rse/prog/Dataset/Visual/VP001/'
CONF_FILE = '/home/rse/prog/Code/DINOv/configs/dinov_sam_coco_swinl_train.yaml'
DINO_CKPT = '/home/rse/prog/Code/DINOv/models/model_swinL.pth'
SAM_CKPT = '/home/rse/prog/Code/DINOv/models/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'

DEVICE = 'cuda'

ref_image_file = REF_FOlDER + 'can_1.jpg'
target_image_file = TEST_FOLDER + '1.jpg'

gray_img, ref_img = hf.img_gray(ref_image_file)
x, y, w, h = hf.draw_bbox_with_contours(gray_img=gray_img, thresh=50)
# cv2.rectangle(ref_img, (x,y), (x+w, y+h), (0,255,0), 2)

# cv2.imshow('bbox', ref_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT)
sam.to(device=DEVICE)

predictor = SamPredictor(sam)
predictor.set_image(ref_img)

input_box = np.array([x, y, x + w, y + h])
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

mask_img = hf.show_mask_opencv(masks[0])

generic_vp1 = {}
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_img = Image.fromarray(ref_img)

mask_img = Image.fromarray(mask_img).convert('RGB')

generic_vp1['image'] = ref_img
generic_vp1['mask'] = mask_img

target_img = Image.open(target_image_file)

opt = load_opt_from_config_file(CONF_FILE)

dino = BaseModel(opt, build_model(opt)).from_pretrained(DINO_CKPT).eval().cuda()

@torch.no_grad()
def inference(generic_vp1, image2, *args, **kwargs):
     with torch.autocast(device_type='cuda', dtype=torch.float16):
        model=dino
        a= task.task_openset(model, generic_vp1, None, None, None,
                   None, None, None, None, image2, *args, **kwargs)
        img = Image.fromarray(a)
        img.save(f'/home/rse/prog/Code/DINOv/test_results/' + 'test_3.jpg')
        return a

inference(generic_vp1,target_img)