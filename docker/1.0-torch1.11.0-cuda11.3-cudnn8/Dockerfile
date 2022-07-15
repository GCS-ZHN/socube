FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install socube==1.0 lapjv==1.3.22 numpy==1.22.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT [ "/opt/conda/bin/socube" ]
CMD [ "--help" ]
