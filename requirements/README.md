# Container Setup

A Docker container was created for this project. Dependencies are all
baked into the container image. Mainly CUDA RAPIDS and common data 
science libraries. The container is built via GitLab CI.

## Gitlab CI Pipeline

Setup environment variables to support build configurations and deployment.

- Test Code Quality
- Test Code Syntax

## Download Container

```bash
module load singularity
singularity build --sandbox nccs-lcluc docker://nasanccs/nccs-lcluc:2021.10
```

## Interacting with the Container

```bash
singularity shell --nv -B $NOBACKUP:$NOBACKUP nccs-lcluc
```
