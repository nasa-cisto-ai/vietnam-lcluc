# Container Setup

A Docker container was created for this project. Dependencies are all
baked into the container image. Mainly CUDA RAPIDS and common data 
science libraries. The container is built via GitLab CI.

## Container Recipe Build

## Gitlab CI Pipeline

- Test Code Quality
- Test Code Syntax

## Download Container

```bash
module load singularity
singularity build --sandbox rapids docker://nvcr.io/nvidia/rapidsai/rapidsai:21.10-cuda11.2-runtime-centos8


docker pull nvcr.io/nvidia/rapidsai/rapidsai:21.10-cuda11.2-base-ubuntu20.04
```
