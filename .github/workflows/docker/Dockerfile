FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
RUN apt-get update && \
  apt-get install -y python3 python3-pip curl python-is-python3 && \
  apt-get -y clean && \
  rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH $PATH:/root/.local/bin
COPY ../../../../hydranet $WORKDIR
WORKDIR ../../../../hydranet
RUN poetry install