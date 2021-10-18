#---------------------------------------------------------------------------------
# Build:
#     docker build --tag helio-hackweek2020:1.0 .
#
# Packages:
#     base: rapids ai cuda 11.2 container
#     additional packages: 
#         TensorFlow, RAPIDS, OpenCV, Rasterio
# Usage:
#     A conda environment called "rapids" has been configured inside the container.
#     "source activate rapids" to activate the environment.
#     
# Date: 10/18/2021 
# Author: Jordan A. Caraballo Vega, jorda.a.caraballo-vega@nasa.gov
#---------------------------------------------------------------------------------

# Arguments to pass to the image
ARG CUDA_VER=11.2
ARG LINUX_VER=ubuntu20.04
ARG FROM_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai

# Import RAPIDS container as the BASE Image (cuda base image)
# Ubuntu 20.04 image with CUDA 11.2 support
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-base-${LINUX_VER}

# Adding ENV variables
ENV RAPIDS_DIR=/rapids

# Install Jax with CUDA support.
RUN source activate rapids \
    && pip install --upgrade rasterio \
    && pip install --upgrade tensorflow tensorflow-addons \
    && pip install --upgrade opencv-python