# ComfyUI-Bagel

ComfyUI-Bagel is now available in ComfyUI, [BAGEL](https://github.com/ByteDance-Seed/Bagel) is an open‑source multimodal foundation model with 7B active parameters (14B total) trained on large‑scale interleaved multimodal data.



## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-Bagel.git
```

3. Install dependencies:
```
cd ComfyUI-Bagel
pip install -r requirements.txt
pip install --no-build-isolation flash-attn
```

## Model

Download models from [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)



## Workflow
1. Download [workflow](https://github.com/Xiuzhenpeng/ComfyUI-Bagel/blob/main/workflow/Bagel.json) and Load in ComfyUI

2. Change the path in node named "Load BAGEL Model" to your folder global path
   ![Example Image](images/Replace_path.png)
