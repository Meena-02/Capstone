# Installation of YOLO-World

1. Creating conda env
```
conda create -n capstone python=3.9 -y
```
2. Install pytorch according to your computer requirements
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
3. Downgrade numpy 
```
pip install numpy==1.26.2
```

4. Install repo
```
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
pip install -e .
```

5. After installing the repo, torch and mmcv will be upgraded to latest version. So need to downgrade torch and mmcv
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

6. Use "https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip" to install the correct mmcv version according your system
```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

7. After installing you will get the following error, ignore it
```
'yolo-world 0.1.0 requires torchvision>=0.16.2, but you have torchvision 0.12.0+cu113 which is incompatible.'
``` 

8. Install other pkgs
```
sudo apt-get install libgl1-mesa-glx
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install gradio
```

# Changes to the YOLO-World-Image Code

1. Download the image prompt yolo model from "https://huggingface.co/spaces/wondervictor/YOLO-World-Image/tree/main". The model file available on the website in under weights. 

2. Download the model file to your folder and save it under model folder

3. Open the file: \
 <b> YOLO-World/configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py </b>

Change line 19:
```
text_model_name = '../pretrained_models/open-ai-clip-vit-base-patch32'
```
to 

```
text_model_name = "openai/clip-vit-base-patch32"
```

4. Open the file: \
 <b> YOLO-World/demo/image_prompt_demo.py </b>

Change line 302:
```
clip_model = "/group/40034/adriancheng/pretrained_models/open-ai-clip-vit-base-patch32"
```
to 
```
clip_model = "openai/clip-vit-base-patch32"
```

5. Open the file: \
<b> YOLO-World/yolo_world/models/detectors/yolo_world_image.py </b>

Add the following in line 241:
```
texts = None
```

# Run the Image Prompt Demo file

To run the image_prompt_demo.py:
```
python demo/image_prompt_demo.py  configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py  model/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth
```