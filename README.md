# SoCube
![license](https://img.shields.io/badge/license-MIT%20License-blue.svg)
![python](https://img.shields.io/badge/python->=3.7-success.svg)
![torch](https://img.shields.io/badge/torch->=1.8.1-success.svg)
[![docker](https://img.shields.io/badge/docker-support-success.svg)](https://hub.docker.com/r/gcszhn/socube)
[![pypi](https://img.shields.io/badge/pypi-release-blue.svg)](https://pypi.org/project/socube/)
[![pmid](https://img.shields.io/badge/PMID-NOT%20available-red.svg)](https://pubmed.ncbi.nlm.nih.gov/)
<img src="fig/workflow.svg" alt="SoCube Workflow">

## Introduction
SoCube was a end-to-end doublet detection tools with novel feature embedding strategy. [User manual](https://www.gcszhn.top/socube/) is published on this repo's github page.

## 1. Installment

### Requirement
SoCube was developed in Python 3.8.11. All third-part requirements are listed in **requirements.txt**。

We recommend to download torch from [pytorch official site](https://pytorch.org/get-started/locally/) or specific pip source of pytorch official. GPU version is recommended if gpu is available. Download from other mirror site maybe a CPU-only version (`torch.cuda.is_available()`will be` False`).
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
### 1.1 As a python package
You can install as python package with following steps.
```bash
git clone https://github.com/GCS-ZHN/socube.git
cd SoCube
conda create -n socube python=3.8.11
conda activate socube
pip install -e . --no-binary lapjv
```
Or, install it from pypi.org
```
conda create -n socube python=3.8.11
conda activate socube
pip install socube --no-binary lapjv
```

### 1.2 As a docker container
- Pull docker image
SoCube published docker images in [dockerhub](https://hub.docker.com/r/gcszhn/socube), you could just pull and use it.
```bash
sudo docker pull gcszhn/socube:latest
```
Besides, you can build a docker container by yourself
```bash
git clone https://github.com/GCS-ZHN/socube.git
cd SoCube
sudo docker build docker -t gcszhn/socube
```
- Create a docker instance

```bash
sudo docker run \
        --gpus all \
        -v <Your input data directory absoluate path outside>:/workspace/datasets \
        gcszhn/socube:latest \
        -i datasets/<Your input data file base name> \
        --gpu-ids 0
```
Please do NOT forget to create container with volume map args (-v), because your input data and output result are hoped to be in your host machine outside of container.

### 1.3 Just use it without installing to python
You can download source code and install requirements and use it
```bash
git clone https://github.com/GCS-ZHN/socube.git
cd SoCube
# conda virtual environment is advised inplace of base env
conda create -n socube python=3.8.11
conda activate socube
pip install -r requirements.txt --no-binary lapjv
```
## 2. Usage
Please visit our user manual for detail.