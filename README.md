# Real-Time Voice Cloning
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time. This is reimplementation of an existing work [CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

SV2TTS is a deep learning framework in three stages. In the first stage, one creates a digital representation of a voice from a few seconds of audio. In the second and third stages, this representation is used as reference to generate speech given arbitrary text.

## Setup

### 1. Install Requirements
1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.9 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. (Optional) Download Pretrained Models
Pretrained models are now downloaded automatically. If this doesn't work for you, you can manually download them [here](https://drive.google.com/drive/folders/1RTg0ePScuhNBihcefiO8EEmtINQNVM_j?usp=sharing).

### 3. (Optional) Test Configuration
Before you download any dataset, you can begin by testing your configuration with:

`python demo_cli.py`

If all tests pass, you're good to go.

### 4. (Optional) Download Datasets
- [`LibriSpeech`](https://www.openslr.org/12),
- [`VCTK`](https://datashare.ed.ac.uk/handle/10283/2950),
- [`UASpeech`](https://ieee-dataport.org/documents/uaspeech),
- [`RAVDESS`](https://zenodo.org/records/1188976).

Extract the contents as `<datasets_root>/<dataset_content>` where `<datasets_root>` is a directory of your choosing. 

### 5. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`  
or  
`python demo_toolbox.py`  
