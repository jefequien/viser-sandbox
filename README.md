![viser_sandbox](assets/viser_sandbox.png)

# Installation

## Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

## Create environment

`viser_sandbox` requires `python >= 3.10`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name viser_sandbox -y python=3.10
conda activate viser_sandbox
pip install --upgrade pip setuptools
```

## Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 12.1).
```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```


### Installing viser_sandbox

Install from pip

```bash
pip install viser_sandbox
```

**OR** install from source.

```bash
git clone https://github.com/jefequien/viser-sandbox.git
cd viser_sandbox
pip install -e .
```
