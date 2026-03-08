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

Create video:
```console
python -m scripts.create_video --config-file data/00_geometric_progression/geometric_progression.json --output-path output/geometric_progression
```