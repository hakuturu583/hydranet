# Hydranet

Hydranet implementation with pytorch

[![build and push docker](https://github.com/hakuturu583/hydranet/actions/workflows/build_docker.yaml/badge.svg)](https://github.com/hakuturu583/hydranet/actions/workflows/build_docker.yaml)

## Setup
### Using with docker

with gpu
```bash
docker run -it --gpus all ghcr.io/hakuturu583/hydranet:latest
```

without gpu
```bash
docker run -it ghcr.io/hakuturu583/hydranet:latest
```