<div align="center">

<!-- TITLE -->
# 🌟**VADER-VideoCrafter**
</div>



We **highly recommend** proceeding with the VADER-VideoCrafter model first, which performs better than the other two.

## ⚙️ Installation
Assuming you are in the `VADER/` directory, you are able to create a Conda environments for VADER-VideoCrafter using the following commands:
```bash
cd VADER-VideoCrafter
conda create -n vader_videocrafter python=3.10
conda activate vader_videocrafter
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..
```


- We are using the pretrained Text-to-Video [VideoCrafter2](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) model via Hugging Face. If you unfortunately find the model is not automatically downloaded when you running inference or training script, you can manually download it and put the `model.ckpt` in `VADER/VADER-VideoCrafter/checkpoints/base_512_v2/model.ckpt`.


## 📺 Inference
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md).

Assuming you are in the `VADER/` directory, you are able to do inference using the following commands:
```bash
cd VADER-VideoCrafter
sh scripts/run_text2video_inference.sh
```
- We have tested on PyTorch 2.3.0 and CUDA 12.1. The inferece script works on a single GPU with 16GBs VRAM, when we set `val_batch_size=1` and use `fp16` mixed precision. It should also work with recent PyTorch and CUDA versions.
- `VADER/VADER-VideoCrafter/scripts/main/train_t2v_lora.py` is a script for inference of the VideoCrafter2 using VADER via LoRA.
    - Most of the arguments are the same as the training process. The main difference is that `--inference_only` should be set to `True`.
    - `--lora_ckpt_path` is required to set to the path of the pretrained LoRA model. Otherwise, the original VideoCrafter model will be used for inference.

## 🔧 Training
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md).

Assuming you are in the `VADER/` directory, you are able to train the model using the following commands:

```bash
cd VADER-VideoCrafter
sh scripts/run_text2video_train.sh
```
- Our experiments are conducted on PyTorch 2.3.0 and CUDA 12.1 while using 4 A6000s (48GB RAM). It should also work with recent PyTorch and CUDA versions. The training script have been tested on a single GPU with 16GBs VRAM, when we set `train_batch_size=1 val_batch_size=1` and use `fp16` mixed precision.
- `VADER/VADER-VideoCrafter/scripts/main/train_t2v_lora.py` is also a script for fine-tuning the VideoCrafter2 using VADER via LoRA.
    - You can read the VADER-VideoCrafter [documentation](../documentation/VADER-VideoCrafter.md) to understand the usage of arguments.


## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.
