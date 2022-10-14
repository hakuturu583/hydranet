FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
RUN apt-get update && \
  apt-get install -y python3-pip wget unzip && \
  apt-get -y clean && \
  rm -rf /var/lib/apt/lists/*
# download torch2trt
RUN wget https://github.com/NVIDIA-AI-IOT/torch2trt/archive/refs/tags/v0.4.0.zip
RUN unzip v0.4.0.zip
# install torch2trt
WORKDIR torch2trt-0.4.0
RUN python3 -m pip install nvidia-pyindex packaging
RUN python3 -m pip install --upgrade nvidia-tensorrt
RUN python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 setup.py install
# install hydranet
WORKDIR ../
COPY *.whl .
RUN pip install *.whl
COPY to_trt.py .