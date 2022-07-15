# SoCube
![license](https://img.shields.io/badge/license-MIT%20License-blue.svg)
![python](https://img.shields.io/badge/python->=3.7-success.svg)
![torch](https://img.shields.io/badge/torch->=1.8.1-success.svg)

<img src="https://github.com/idrblab/SoCube/blob/master/fig/Figure%201.svg" alt="SoCube Workflow">

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
git clone https://github.com/GCS-ZHN/SoCube.git
cd SoCube
conda create -n socube python=3.8.11
conda activate socube
pip install -e . --no-binary lapjv
```
Or, install it from pypi.org
```
conda create -n socube python=3.8.11
conda activate socube
pip install socube --no-binary lapjv -i https://pypi.org/simple/
```

### 1.2 As a docker container
- Pull docker image
SoCube published docker images in [dockerhub](https://hub.docker.com/r/gcszhn/socube), you could just pull and use it.
```bash
sudo docker pull gcszhn/socube:latest
```
Besides, you can build a docker container by yourself
```bash
git clone https://github.com/GCS-ZHN/SoCube.git
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

If you want to use docker under windows 10, please make sure the version number of win10 is up to a newer version, such as 21H2. Otherwise it may lead to GPU unavailability with an error similar to the following: "Running hook #0:: error running hook: exit status 1, stdout: , stderr: nvidia-container-cli: initialization error: driver error: failed to process request: unknown". Or you can just use cpu as follow:
```bash
docker run \
        -v <Your input data directory absoluate path outside>:/workspace/datasets \
        gcszhn/socube:latest \
        -i datasets/<Your input data file base name> 
```

### 1.3 Just use it without installing to python
You can download source code and install requirements and use it
```bash
git clone https://github.com/GCS-ZHN/SoCube.git
cd SoCube
# conda virtual environment is advised inplace of base env
conda create -n socube python=3.8.11
conda activate socube
pip install -r requirements.txt --no-binary lapjv
```
## 2. Usage
### Local usage
A h5ad format (anndata) or h5 format (pandas DataFrame) is required as input format of scRNA-seq data. For simple usage, just like following in local terminal. We recommend to use `--enable-multiprocess` to enable multiprocess for parallel training, giving full play to the performance advantages of modern multi-core CPUs to improve training speed. Also, if you have multiple gpu's, you can use `--gpu-ids 0,1,2,3` to specify the specific GPU to use for parallel training.
```bash
socube -i your_sc.h5ad -o your_sc --gpu-ids 0 --enable-multiprocess
```
For detail, just type `socube` or `socube -h` to get help details.  You can also run it as an executable python package `python -m socube`.

### Colab usage
Google providers a online machine learning platform [colab](https://colab.research.google.com/) with GPU for free. You can upload `socube_colab.ipynb` (provided by us in this repository) and scRNA-seq data (.h5ad) to your [google drive](https://drive.google.com/). Please learn how to use colab's gpu by yourself.

## 3. Notice
### 3.1 About third-part library "lapjv"
You need to pre-install numpy before installing some versions of lapjv dependencies (otherwise you will see the error ModuleNotFoundError: No module named 'numpy'), this is because these versions, such as v1.3.1, directly `import numpy` in `setup.py`.

Also the source installation needs to provide cpp library dependencies, for windows, you can install the full visual studio 2019 or just download the build tool: https://visualstudio.microsoft.com/visual-cpp-build-tools/. The download and installation is complete and available after restarting the computer.

Installing the binary version (wheel) of lapjv, the following `RuntimeError` may exist when using it. This is because the C library API version of numpy used by the publisher to compile lapjv is different from the currently installed numpy.
```
RuntimeError: module compiled against API version 0xf but this version of numpy is 0xe
```
There are two ways to solve this, one is to install lapjv in source code, which requires a newer version of gcc with `-std=c++17` support.
```
pip install lapjv --no-binary lapjv
```
The second is to install the corresponding version of numpy, although this may have a dependency conflict.

For other questions about lapjv, please go to the official repository [src-d/lapjv](https://github.com/src-d/lapjv).

### 3.2 About third-part library "pytables"
Some version of this packge missing required dynamic C library, such as `tables-3.7.0-cp38-cp38-win_amd64`, you solve it by trying to install another version listed in [PyPi](https://pypi.org/project/tables/). Any other question, you can see its [official GitHub repository](https://github.com/PyTables/PyTables).
```
Traceback (most recent call last):
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\pandas\compat\_optional.py", line 138, in import_optional_dependency
    module = importlib.import_module(name)
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\importlib\__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 843, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\tables\__init__.py", line 45, in <module>
    from .utilsextension import get_hdf5_version as _get_hdf5_version
ImportError: DLL load failed while importing utilsextension: 找不到指定的模块。

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\concurrent\futures\process.py", line 239, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "d:\life_matters\IDRB\深度组学\单细胞组学\SoCube\src\socube\utils\concurrence.py", line 74, in wrapper
    return func(*args, **kwargs)
  File "d:\life_matters\IDRB\深度组学\单细胞组学\SoCube\src\socube\utils\io.py", line 262, in writeHdf
    data.to_hdf(file, key=key, mode=mode, **kwargs)
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\pandas\core\generic.py", line 2763, in to_hdf
    pytables.to_hdf(
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\pandas\io\pytables.py", line 311, in to_hdf
    with HDFStore(
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\pandas\io\pytables.py", line 572, in __init__
    tables = import_optional_dependency("tables")
  File "c:\Users\zhang\anaconda3\envs\socube_test\lib\site-packages\pandas\compat\_optional.py", line 141, in import_optional_dependency
    raise ImportError(msg)
ImportError: Missing optional dependency 'pytables'.  Use pip or conda to install pytables.
```