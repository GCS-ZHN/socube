# SoCube
![license](https://img.shields.io/badge/license-MIT%20License-blue.svg)
![python](https://img.shields.io/badge/python->=3.7-success.svg)
![torch](https://img.shields.io/badge/torch->=1.8.1-success.svg)
[![docker](https://img.shields.io/badge/docker-support-success.svg)](https://hub.docker.com/r/gcszhn/socube)
[![pypi](https://img.shields.io/badge/pypi-release-blue.svg)](https://pypi.org/project/socube/)
[![pmid](https://img.shields.io/badge/PMID-NOT%20available-red.svg)](https://pubmed.ncbi.nlm.nih.gov/)
<img src="fig/workflow.svg" alt="SoCube Workflow">

## Introduction
SoCube was a end-to-end doublet detection tool with novel feature embedding strategy. [User manual](https://www.gcszhn.top/socube/) is published on this repo's github page.

## Installment
You can just install socube by execute pip command. 
```bash
pip install socube
```
For install detail, please see at [user manual](https://www.gcszhn.top/socube/).
## Usage
Basic usage of socube with gpus is like following example.
```bash
socube -i data/pbmc-1C-dm.h5ad --gpu-ids 0
```
Please visit our [user manual](https://www.gcszhn.top/socube/) for usage detail.

## Help
Any problem, you could create an [issue](https://github.com/GCS-ZHN/socube/issues), we will receive a email sented by github and reply it as soon as possible.