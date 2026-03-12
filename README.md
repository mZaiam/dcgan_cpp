# DCGAN in C++

A implementation of a Deep Convolutional Generative Adversarial Network [(DCGAN)](https://arxiv.org/abs/1511.06434) written in C++. This project was done towards learning C++ and some of its tools. It trains a Generator and Discriminator based on the MNIST dataset on CPUs.

## Results

![Generated MNIST Digits](checkpoints/inference-sample-epoch-30-batch-900.png) 

## Prerequisites

To build and run this project, your system needs:
* **C++ Compiler** supporting C++20 (GCC/Clang)
* **CMake** (>= 3.10)
* **LibTorch** (The PyTorch C++ API)
* **OpenCV** (For saving inference tensors as `.png` files)

## Dataset

LibTorch requires the uncompressed, raw MNIST IDX files. Create a `data/` directory and run the `mnist.py` file for dowloading the files:
