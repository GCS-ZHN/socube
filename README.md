# SoCube
![license](https://img.shields.io/badge/license-MIT%20License-blue.svg)
![python](https://img.shields.io/badge/python->=3.7-success.svg)
![torch](https://img.shields.io/badge/torch->=1.8.1-success.svg)
[![docker](https://img.shields.io/badge/docker-support-success.svg)](https://hub.docker.com/r/gcszhn/socube)
[![pypi](https://github.com/GCS-ZHN/socube/actions/workflows/pypi.yml/badge.svg)](https://pypi.org/project/socube/)
[![pmid](https://img.shields.io/badge/PMID-36941114-blue.svg)](https://pubmed.ncbi.nlm.nih.gov/36941114/)
<img src="fig/workflow.svg" alt="SoCube Workflow">

## Introduction
SoCube is an end-to-end doublet detection tool with novel feature embedding strategy. [User manual](https://www.gcszhn.top/socube/) is published on this repo's github page.

## Installment
You can just install socube by executing pip command. 
```bash
pip install socube
```
For install detail, please see at [user manual](https://www.gcszhn.top/socube/).
## Usage
The basic usage of socube with gpus is as follows.
```bash
socube -i data/pbmc-1C-dm.h5ad --gpu-ids 0
```
Please visit our [user manual](https://www.gcszhn.top/socube/) for usage detail.

## Paper reproduce
This repo is open source for socube software. If you want reprocduce result in original paper, Please visit repo [GCS-ZHN/socube-reproduce](https://github.com/GCS-ZHN/socube-reproduce/).

## Help
Any problem, you could create an [issue](https://github.com/GCS-ZHN/socube/issues), we will receive an email sent by github and reply it as soon as possible.

## Citation
If you used our software, please cite it.
```bibtex
@article{10.1093/bib/bbad104,
    author = {Zhang, Hongning and Lu, Mingkun and Lin, Gaole and Zheng, Lingyan and Zhang, Wei and Xu, Zhijian and Zhu, Feng},
    title = "{SoCube: an innovative end-to-end doublet detection algorithm for analyzing scRNA-seq data}",
    journal = {Briefings in Bioinformatics},
    year = {2023},
    month = {03},
    abstract = "{Doublets formed during single-cell RNA sequencing (scRNA-seq) severely affect downstream studies, such as differentially expressed gene analysis and cell trajectory inference, and limit the cellular throughput of scRNA-seq. Several doublet detection algorithms are currently available, but their generalization performance could be further improved due to the lack of effective feature-embedding strategies with suitable model architectures. Therefore, SoCube, a novel deep learning algorithm, was developed to precisely detect doublets in various types of scRNA-seq data. SoCube (i) proposed a novel 3D composite feature-embedding strategy that embedded latent gene information and (ii) constructed a multikernel, multichannel CNN-ensembled architecture in conjunction with the feature-embedding strategy. With its excellent performance on benchmark evaluation and several downstream tasks, it is expected to be a powerful algorithm to detect and remove doublets in scRNA-seq data. SoCube is freely provided as an end-to-end tool on the Python official package site PyPi (https://pypi.org/project/socube/) and open-source on GitHub (https://github.com/idrblab/socube/).}",
    issn = {1477-4054},
    doi = {10.1093/bib/bbad104},
    url = {https://doi.org/10.1093/bib/bbad104},
    note = {bbad104},
    eprint = {https://academic.oup.com/bib/advance-article-pdf/doi/10.1093/bib/bbad104/49570241/bbad104.pdf},
}
```
