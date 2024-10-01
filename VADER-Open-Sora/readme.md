<div align="center">

<!-- TITLE -->
# 🎬 **VADER-Open-Sora**

</div>


## ⚙️ Installation
Assuming you are in the `VADER/` directory, you are able to create a Conda environments for VADER-Open-Sora using the following commands:
```bash
cd VADER-Open-Sora
conda create -n vader_opensora python=3.10
conda activate vader_opensora
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -v -e .
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2/
pip install -e .
cd ..
```

## 📺 Inference
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md).

Assuming you are in the `VADER/` directory, you are able to do inference using the following commands:
```bash
cd VADER-Open-Sora
sh scripts/run_text2video_inference.sh
```
- We have tested on PyTorch 2.3.0 and CUDA 12.1. If the `resolution` is set as `360p`, a GPU with 40GBs of VRAM is required when we set `val_batch_size=1` and use `bf16` mixed precision . It should also work with recent PyTorch and CUDA versions. Please refer to the original [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository for more details about the GPU requirements and the model settings.
- `VADER/VADER-Open-Sora/scripts/train_t2v_lora.py` is a script for do inference via the Open-Sora 1.2 using VADER.
    - `--num-frames`, `'--resolution'`, `'fps'` and `'aspect-ratio'` are inherited from the original Open-Sora model. In short, you can set `'--num-frames'` as `'2s'`, `'4s'`, `'8s'`, and `'16s'`. Available values for `--resolution` are `'240p'`, `'360p'`, `'480p'`, and `'720p'`. The default value of `'fps'` is `24` and `'aspect-ratio'` is `3:4`. Please refer to the original [Open-Sora](https://github.com/hpcaitech/Open-Sora) repository for more details. One thing to keep in mind, for instance, is that if you set `--num-frames` to `2s` and `--resolution` to `'240p'`, it is better to use `bf16` mixed precision instead of `fp16`. Otherwise, the model may generate noise videos.
    - `--prompt-path` is the path of the prompt file. Unlike VideoCrafter, we do not provide prompt function for Open-Sora. Instead, you can provide a prompt file, which contains a list of prompts.
    - `--num-processes` is the number of processes for Accelerator. It is recommended to set it to the number of GPUs.
- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_inferece.py` is the configuration file for inference. You can modify the configuration file to change the inference settings following the guidance in the [documentation](../documentation/VADER-Open-Sora.md).
    - The main difference is that `is_vader_training` should be set to `False`. The `--lora_ckpt_path` should be set to the path of the pretrained LoRA model. Otherwise, the original Open-Sora model will be used for inference.


## 🔧 Training
Please run `accelerate config` as the first step to configure accelerator settings. If you are not familiar with the accelerator configuration, you can refer to VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md).

Assuming you are in the `VADER/` directory, you are able to train the model using the following commands:

```bash
cd VADER-Open-Sora
sh scripts/run_text2video_train.sh
```
- Our experiments are conducted on PyTorch 2.3.0 and CUDA 12.1 while using 4 A6000s (48GB RAM). It should also work with recent PyTorch and CUDA versions. A GPU with 48GBs of VRAM is required for fine-tuning model when use `bf16` mixed precision as `resolution` is set as `360p` and `num_frames` is set as `2s`.
- `VADER/VADER-Open-Sora/scripts/train_t2v_lora.py` is a script for fine-tuning the Open-Sora 1.2 using VADER via LoRA.
    - The arguments are the same as the inference process above.
- `VADER/VADER-Open-Sora/configs/opensora-v1-2/vader/vader_train.py` is the configuration file for training. You can modify the configuration file to change the training settings.
    - You can read the VADER-Open-Sora [documentation](../documentation/VADER-Open-Sora.md) to understand the usage of arguments.




## Acknowledgement

Our codebase is directly built on top of [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter), [Open-Sora](https://github.com/hpcaitech/Open-Sora), and [Animate Anything](https://github.com/alibaba/animate-anything/). We would like to thank the authors for open-sourcing their code.
