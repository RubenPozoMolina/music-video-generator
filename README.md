# music-video-generator

Application to generate music videos

## Requirements

The application has been tested only with CUDA.

## Installation

Create virtual environment
```console
python3 -m venv .venv
source .venv/bin/activate
```

Install torch with cuda:
```console
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```
Nightly version of torch:
```console
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 
``` 

Alternatively, you can verify that the installation was successful by running:
```console
python utils/verify.py
```

If you encounter any issues during installation, please refer to the [official PyTorch documentation](https://pytorch.org/get-started/locally/) for more detailed instructions.

Install requirements:
```console
pip install -r requirements.txt
```

## Usage

Create image:
```console
python -m scripts.create_image
```

Modify image
```console
python -m scripts.modify_image
```

Create video from screenplay
```console
python -m scripts.create_video_from_screenplay
```