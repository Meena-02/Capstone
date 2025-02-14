import os
import tqdm
import argparse
import os.path as osp
import numpy as np
from PIL import Image
from transformers import (AutoTokenizer, AutoProcessor,
                          CLIPVisionModelWithProjection,
                          CLIPTextModelWithProjection)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default="openai/clip-vit-base-patch32")
        # default='../pretrained_models/open-ai-clip-vit-base-patch32')
    parser.add_argument('--image-dir', type=str, default='data/samples.txt')
    parser.add_argument('--out-dir', type=str, default='YOLO-World/data/image_prompts/')
    parser.add_argument('--out-file', type=str)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(args.model)
    text_model = CLIPTextModelWithProjection.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    # padding prompts
    device = 'cuda:0'
    text_model.to(device)
    texts = tokenizer(text=[' '], return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = text_model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1]).cpu().data.numpy()
    txt_feats = np.squeeze(txt_feats)  # Remove unnecessary dimensions
    # print(f"Text embedding shape: {txt_feats.shape}")
    
    images = os.listdir(args.image_dir)
    category_embeds = []

    def _forward_vision_model(image_name):
        image_path = osp.join(args.image_dir, image_name)
        # category = image_name.split('-')[1]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_outputs = vision_model(**inputs)
        img_feats = image_outputs.image_embeds
        # img_feats
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        img_feats = img_feats.reshape(
            -1, img_feats.shape[-1])[0].cpu().data.numpy()
        # print(f"Image {image_name} embedding shape: {img_feats.shape}")
        category_embeds.append(img_feats)

    for image_ in tqdm.tqdm(images):
        _forward_vision_model(image_)
    category_embeds.append(txt_feats)
    # for i, emb in enumerate(category_embeds):
    #     print(f"Embedding {i}: shape {emb.shape}")
    category_embeds = np.stack(category_embeds)
    np.save(osp.join(args.out_dir, args.out_file), category_embeds)
