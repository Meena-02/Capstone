import gradio as gr
import video_hf as hf
from PIL import Image, ImageEnhance
import os
import cv2
import threading
import time

from paddleocr import PaddleOCR, draw_ocr

from functools import partial 
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmengine.dataset import Compose
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)

# Global variables to control the live stream
live_stream_active = False # Flag to indicate if live stream is running
current_frame = None # Stores the latest webcam frame for display

def demo(runner, vision_encoder, vision_processor, padding_embed, text_model):
    """Creates the Gradio UI for video processing with live streaming support."""
    with gr.Blocks() as demo:
        gr.Markdown("<center><h1>Capstone Demo</h1></center>")
        
        with gr.Row():
            with gr.Column():
                # Radio button to select between live stream and upload video
                video_source = gr.Radio(["Live Stream", "Upload Video"], label="Video Source", value="Upload Video")
                target_video = gr.Video(label="Target Video", visible=False)
                webcam_feed = gr.Image(label="Live Stream", visible=False)
                
                # Tabs for reference images
                with gr.TabItem("Image 1"):
                    ref_image1 = gr.Image(label='Reference Image', type='pil', height=300)
                with gr.TabItem("Image 2"):
                    ref_image2 = gr.Image(label="Reference Image", type='pil', height=300)
                
                # Buttons for live stream and video processing
                with gr.Row():
                    start_button = gr.Button("Start Live Stream")
                    stop_button = gr.Button("Stop Live Stream")
                    submit_button = gr.Button("Submit")
                    clear_button = gr.Button("Clear")
            
            with gr.Column():
                output_video = gr.Video(label="Output Video") # Displays the processed video
                input_text = gr.Textbox(label="Text Prompt") # text input for matching ocr results
                
        def validate_and_run(video_source, target_video, input_text, ref_image1, ref_image2):
            """Handles the start of video processing or live stream."""
            global live_stream_active
            
            if video_source == "Live Stream":
                live_stream_active = True
                threading.Thread(target=live_stream, args=(runner, vision_encoder, vision_processor, padding_embed, text_model, input_text, ref_image1, ref_image2)).start()
                return None  # No output video for live stream
            else:
                live_stream_active = False
                if target_video is None:
                    raise gr.Error("Please upload target video")
            
            if not input_text.strip():
                raise gr.Error("Please enter a text prompt")
            if ref_image1 is None or ref_image2 is None:
                raise gr.Error("Please upload both 2 reference images")
            
            # Enhancing reference images for better object detection
            prompt_images = [ref_image1, ref_image2]
            enhancer = ImageEnhance.Contrast(ref_image2)
            prompt_images.append(enhancer.enhance(1.5))
            
            # Run video processing function
            output_video = hf.run_video(runner, vision_encoder, vision_processor, padding_embed, text_model, target_video, input_text, prompt_images)
            return output_video
        
        def live_stream(runner, vision_encoder, vision_processor, padding_embed, text_model, input_text, ref_image1, ref_image2):
            """Handles real-time webcam inference."""
            global live_stream_active, current_frame
            
            cap = cv2.VideoCapture(0)  # Open the webcam
            if not cap.isOpened():
                raise gr.Error("Error opening webcam")
            
            while live_stream_active:
                ret, frame = cap.read()
                if not ret or not live_stream_active:
                    break # Stop streaming if the flag is turned off
                
                # Convert opencv frames to PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                try:
                    # Run object detection on the frame
                    detected_object = hf.run_image2(
                        runner, vision_encoder, vision_processor, padding_embed, text_model, frame_pil, input_text, [ref_image1, ref_image2]
                    )
                    if detected_object is not None:
                        bbox = detected_object.get('coord')
                        if bbox is not None:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw bounding box
                except Exception as e:
                    print(f"Error during inference: {e}")
                
                # Update the global current frame to be displayed
                current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time.sleep(0.03)
            
            cap.release()
        
        def start_live_stream(input_text, ref_image1, ref_image2):
            """Starts the live stream inference."""
            global live_stream_active
            
            live_stream_active = True
            threading.Thread(target=live_stream, args=(runner, vision_encoder, vision_processor, padding_embed, text_model, input_text, ref_image1, ref_image2)).start()
            return None

        def stop_live_stream():
            """Stops the live stream inference."""
            global live_stream_active
            live_stream_active = False
            return None
                
        def update_webcam_feed():
            """Continuously updates the live webcam feed in Gradio."""
            global current_frame
            while live_stream_active:
                if current_frame is not None:
                    yield current_frame
                time.sleep(0.03)
        
        def update_video_source(video_source):
            """Updates UI visibility based on video source selection."""
            if video_source == "Live Stream":
                return gr.Video(visible=False), gr.Image(visible=True)
            else:
                return gr.Video(visible=True), gr.Image(visible=False)
        
        # Toggle visibility based on user selection
        video_source.change(
            update_video_source,
            inputs=[video_source],
            outputs=[target_video, webcam_feed]
        )
        
        # Connect buttons to their respective functions
        start_button.click(start_live_stream, inputs=[input_text, ref_image1, ref_image2])
        stop_button.click(stop_live_stream)
        
        submit_button.click(
            validate_and_run,
            inputs=[video_source, target_video, input_text, ref_image1, ref_image2],
            outputs=[output_video]
        )
        
        clear_button.click(
            lambda: [None, None, None, '', None], 
            inputs=[],
            outputs=[target_video, ref_image1, ref_image2, input_text, output_video]
        )
        
        # Updates the webcam feed while streaming
        demo.load(update_webcam_feed, None, webcam_feed, every=0.03)
        
        demo.launch()

if __name__ == '__main__':
    """Load models and launch Gradio demo."""
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
    pipeline = visual_cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
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
