#!/usr/bin/env bash
rm /usr/local/cuda
ln -s /usr/local/cuda-10.1 /usr/local/cuda
ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib64-nvidia
nvcc --version