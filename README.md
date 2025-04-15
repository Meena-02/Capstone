# 🛒 Vision-Text Fusion for Automated Grocery Warehousing

This project is a capstone implementation that integrates visual prompt-based object detection with OCR-based text extraction to improve product identification in grocery warehouse environments. The system is designed to help reduce incorrect order picking and inventory errors by leveraging both visual features and textual labels.

---

## 📌 Project Overview

In traditional warehouse systems, visually similar items (e.g., different flavors of the same brand) often lead to mispicks. This project proposes a hybrid model that:
- Detects products using **YOLO-World-Image**, a vision-language model.
- Verifies product identity using **PaddleOCR** to extract and compare text labels.
- Uses fuzzy string matching to validate the OCR text against user input (e.g., product name).
- Supports testing across various challenging scenarios: occlusion, clutter, scale variation, and product disambiguation.

---

## 🧠 Key Features

- 🔎 **Open-vocabulary object detection** with YOLO-World
- 🧾 **Multilingual text extraction** using PaddleOCR
- 🧠 **Vision-text fusion pipeline** for reliable product validation
- 🧪 Evaluation on 5 diverse datasets (VP001–VP005)
- 🖼️ Gradio-based UI for image input and result visualization

---

## 📂 Project Structure
TBD

---

## ⚙️ Installation

### Installation of YOLO-World
Create conda env
```
conda create -n capstone python=3.9 -y
```
Install pytorch according to computer requirements
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
Downgrade numpy
```
pip install numpy==1.26.2
```
Install repo
```
[git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
pip install -e .](https://github.com/Meena-02/Capstone.git)
cd YOLO-World
pip install -e .
```
After installing the YOLO-World requirements, torch and mmcv will be upgraded to their latest version. Must downgrade torch and mmcv
```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
Use "https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip" to install the correct mmcv version according to computer
```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```
After installing an error will be displayed, ignore it
```
'yolo-world 0.1.0 requires torchvision>=0.16.2, but you have torchvision 0.12.0+cu113 which is incompatible.'
```
Install other packages
```
sudo apt-get install libgl1-mesa-glx
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install gradio
```
---
### Install PaddleOCR
Install paddle with gpu according to your computer settings. (https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)
```
python -m pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
Install paddleocr
```
pip install paddleocr
```

NOTE: AFTER INSTALLING EVERYTHING, SET THE PYTHON PATH TO YOLO-WORLD
```
export PYTHONPATH=~/YOLO-World
```

---
## 🚀 Running the System
OPTION 1: Run only single prompt image
```
python demo.py
```
![image](https://github.com/user-attachments/assets/b6cbf11f-8435-4e15-a37a-5c1019894332)

OPTION 2: Run multiple prompt image
```
python demo_prompts.py
```
OPTION 3: Run live video inference/video inference with multiple prompt
```
python demo_prompts_video.py
```
## 🧪 Evaluation Datasets
The system was tested on five structured datasets:


| Dataset	| Description                       |
|---------|-----------------------------------|
| VP001	  | Control set (white background)    |
| VP002	  | Two products (disambiguation test)|
| VP003	  | Occlusion test                    |
| VP004	  | Misplaced product in clutter      |
| VP005	  | Scale variation (far/close shots) |

Results showed strong accuracy under clean and semi-cluttered conditions, with challenges in occlusion and far-scale images.

## 🧩 Limitations
Fine-tuning of YOLO-World was planned but not completed due to unresolved issues in the GitHub repo.

Robotic arm integration was out of scope due to time and environment constraints.

Performance drops in cases of severe occlusion or low-resolution text.

## 🔮 Future Work
Fine-tune YOLO-World with custom grocery datasets

Integrate the pipeline with a real/simulated robotic arm

Improve OCR accuracy using text enhancement or fine-tuned models

Add multilingual text support and dynamic thresholding

Deploy the system on an edge device using ONNX/TensorRT
