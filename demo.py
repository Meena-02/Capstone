import gradio as gr
import cv2

def inputs(target_image, ref_image, text, max_box, nms, score):
    print(f"Target image: {type(target_image)}")
    print(f"Ref image: {type(ref_image)}")
    print(f"Text: {text}")
    print(f"Max Boxes: {max_box}")
    print(f"NMS: {nms}")
    print(f"Score: {score}")
    
    return target_image

def clear_inputs():
    return None, None, None, "", 100, 0.7, 0.3

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Capstone Demo</h1></center>")
    with gr.Row():
        with gr.Column():
            target_image = gr.Image(label="Target Image")
            ref_image = gr.Image(label="Reference Image")
            with gr.Row():
                submit_button = gr.Button("Submit")
                clear_button = gr.Button("Clear")
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            input_text = gr.Textbox(label="Text Prompt")
            max_box = gr.Slider(minimum=0, maximum=300, label="Maximum Number of Boxes", step=1, value=100)
            nms = gr.Slider(minimum=0, maximum=1, label="NMS Threshold", step=0.01, value=0.7)
            score = gr.Slider(minimum=0, maximum=1, label="Score Threshold", step=0.01, value=0.3)
    
    submit_button.click(inputs, inputs=[target_image, ref_image, input_text, max_box, nms, score], outputs=[output_image])
    clear_button.click(clear_inputs, inputs=[], outputs=[target_image, ref_image, output_image, input_text, max_box, nms, score])
    
demo.launch()