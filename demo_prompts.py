import gradio as gr
import helper_functions as hf
from PIL import Image, ImageEnhance

from paddleocr import PaddleOCR, draw_ocr

from functools import partial 
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmengine.dataset import Compose
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)

def demo(runner, vision_encoder, vision_processor, padding_embed, text_model):
    with gr.Blocks() as demo:
        gr.Markdown("<center><h1>Capstone Demo</h1></center>")
        
        with gr.Row():
            with gr.Column():
                target_image = gr.Image(label="Target Image", type="pil")
                with gr.TabItem("Image 1"):
                    ref_image1 = gr.Image(label='Reference Image', type='pil', height=300)
                with gr.TabItem("Image 2"):
                    ref_image2 = gr.Image(label="Reference Image", type='pil', height=300)
                with gr.Row():
                    submit_button = gr.Button("Submit")
                    clear_button = gr.Button("Clear")
            with gr.Column():
                output_image = gr.Image(label="Output Image", type="pil")
                input_text = gr.Textbox(label="Text Prompt")
                
        def validate_and_run(target_image, input_text, ref_image1, ref_image2):
            if target_image is None:
                raise gr.Error("Please upload target image")
            if not input_text.strip():
                raise gr.Error("Please enter a text prompt")
            
            prompt_images = []
            enhancer = ImageEnhance.Contrast(ref_image2)
            ref_img3 = enhancer.enhance(1.5)
            for img in [ref_image1, ref_image2, ref_img3]:
                if img is not None:
                    prompt_images.append(img)
            
            if len(prompt_images) == 0:
                raise gr.Error("Please upload atleast 1 reference image")            
            
            output_image = hf.run_image2(runner, vision_encoder, vision_processor, padding_embed, text_model, target_image, input_text, prompt_images)
            return output_image
        
        submit_button.click(
            validate_and_run,
            inputs=[target_image, input_text, ref_image1, ref_image2],
            outputs=[output_image]
        )
        
        clear_button.click(
            lambda: [None, None, None, '', None], 
            inputs=[],
            outputs = [target_image, ref_image1, ref_image2, input_text, output_image])
        
        demo.launch()

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
    
    demo(runner, vision_model, processor, txt_feats, paddle_ocr)
