import sys
import os
from PIL import Image
import evaluation_helper_function as eval
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import jiwer

from paddleocr import PaddleOCR
from functools import partial 
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmengine.dataset import Compose
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)
import torch

torch.cuda.empty_cache()

# INPUT_CSV_FILE = 'Dataset/Visual/test_full_dataset.csv'
# OUTPUT_CSV_FILE = "Dataset/Visual/eval_results_baseline.csv"

INPUT_CSV_FILE = 'Dataset/Visual/test_full_dataset_edited.csv'
OUTPUT_CSV_FILE = "Dataset/Visual/eval_results_trans_bg.csv"

# INPUT_CSV_FILE = 'Dataset/Visual/VP005/test_a.csv'
# OUTPUT_CSV_FILE = 'Dataset/Visual/VP005/eval_results_baseline.csv'

# INPUT_CSV_FILE = 'Dataset/Visual/VP005/test_a_edited.csv'
# OUTPUT_CSV_FILE = 'Dataset/Visual/VP005/eval_results_trans_bg.csv'

df = pd.read_csv(INPUT_CSV_FILE)

results = {
    'image_id': [],
    'iou' : [],
    'vis_success': [],
    'cer': [],
    'wer': [],
    'exact_match': [],
    'fuzzy_score': [],
    'text_success': [],
    'overall_success': []
}

paddle_ocr = PaddleOCR(lang='en')
demo_cfg = eval.read_config("config.yaml")
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

for index, row in df.iterrows():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Target Image: {row['target_img']}")
    print(f"Ref image: {row['ref_img']}")
    print(f"Input text: {row['input_text']}")
    print(f"Annotated image: {row['annotated_img']}")
    print(f"Label file: {row['label_file']}")
    
    target_img = Image.open(row['target_img'])
    ref_img = Image.open(row['ref_img'])
    input_text = row['input_text']
    
    annotated_img = Image.open(row['annotated_img'])
    a_width, a_height = annotated_img.size
    label_file = row['label_file']
    
    yolo_bboxes = eval.get_xyxy_from_yolo(label_file, a_width, a_height)
    final_obj = eval.run_image(runner, vision_model, processor, txt_feats,
                                paddle_ocr, target_img, input_text, ref_img)
            
    
    if yolo_bboxes is None or final_obj is None:
        print(f"Error: Predicted bbox or Ground truth bbox is None. Skipping IOU calculation")
        results['image_id'].append(row['target_img'])
        results['iou'].append(0)
        results['vis_success'].append(0)
        results['cer'].append(0)
        results['wer'].append(0)
        results['exact_match'].append(0)
        results['fuzzy_score'].append(0)
        results['text_success'].append(0)
        results['overall_success'].append(0)
        continue
    
    iou_score = eval.calculate_iou(final_obj['coord'], yolo_bboxes)
    
    extracted_text = " ".join(final_obj['text'])
    ground_truth_text = input_text.lower()
    
    print(f"extracted_text: {extracted_text}")
    print(f"input_text: {input_text}")

    cer = jiwer.cer(ground_truth_text, extracted_text)
    wer = jiwer.wer(ground_truth_text, extracted_text)

    # Compute Text Matching Metrics
    exact_match = int(ground_truth_text == extracted_text)
    fuzzy_score = fuzz.ratio(ground_truth_text, extracted_text)

    # Determine overall success
    overall_success = 1 if iou_score >= 0.5 and fuzzy_score >= 50 else 0
    vis_success = 1 if iou_score >= 0.5 else 0
    text_success = 1 if fuzzy_score >= 50 else 0

    # Store results
    results['image_id'].append(row['target_img'])
    results['iou'].append(iou_score)
    results['vis_success'].append(vis_success)
    results['cer'].append(cer)
    results['wer'].append(wer)
    results['exact_match'].append(exact_match)
    results['fuzzy_score'].append(fuzzy_score)
    results['text_success'].append(text_success)
    results['overall_success'].append(overall_success)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"\nEvaluation complete! Results saved to {OUTPUT_CSV_FILE}")

# Compute Overall Accuracy
overall_accuracy = results_df['overall_success'].mean()
print(f"\nOverall End-to-End System Accuracy: {overall_accuracy:.2%}")

visual_accuracy = results_df['vis_success'].mean()
print(f"\nVisual Accuracy: {visual_accuracy:.2%}")

text_accuracy = results_df['text_success'].mean()
print(f"\nText Accuracy: {overall_accuracy:.2%}")

# Generate Confusion Matrix
y_true = results_df['overall_success']
y_pred = results_df['exact_match']

conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(5, 5))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.show()

# Generate Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Identify failure cases
failure_cases = results_df[results_df['overall_success'] == 0]
print("\nFailure Cases (First 5 rows):")
print(failure_cases.head())

print("\nEvaluation Summary:")
print(results_df.describe())


print("Script completed successfully!")  
os._exit(0) # Explicitly terminate
