# DCGAN in C++

A implementation of a Deep Convolutional Generative Adversarial Network [(DCGAN)](https://arxiv.org/abs/1511.06434) written in C++. This project was done towards learning C++ and some of its tools. It trains a Generator and Discriminator based on the MNIST dataset on CPUs. The code was inspired by the PyTorch tutorial on the C++ framework [PyTorch C++ Tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html).

## Results

As the GAN trains, the Generator learns to map a 100-dimensional latent noise vector into handwritten digits. Below is the progression of the generated samples across training steps:

<div align="center">
  <img src="images/train-sample1.png" width="90" alt="Epoch 1">
  <img src="images/train-sample2.png" width="90" alt="Epoch 2">
  <img src="images/train-sample3.png" width="90" alt="Epoch 3">
  <img src="images/train-sample4.png" width="90" alt="Epoch 4">
  <img src="images/train-sample5.png" width="90" alt="Epoch 5">
  <img src="images/train-sample6.png" width="90" alt="Epoch 6">
  <img src="images/train-sample7.png" width="90" alt="Epoch 7">
  <img src="images/train-sample8.png" width="90" alt="Epoch 8">
  <img src="images/train-sample9.png" width="90" alt="Epoch 9">
  <img src="images/train-sample10.png" width="90" alt="Epoch 10">
</div>
<p align="center">
  <em>Samples from the Generator from Epoch 1 (left) to Epoch 10 (right).</em>
</p>

## Prerequisites

To build and run this project, your system needs:
* **C++ Compiler** supporting C++20 (GCC/Clang)
* **CMake** (>= 3.10)
* **LibTorch** (The PyTorch C++ API)
* **OpenCV** (For saving inference tensors as `.png` files)

## Dataset

LibTorch requires the uncompressed, raw MNIST IDX files. Create a `data/` directory and run the `mnist.py` file for dowloading the files.

## Build 

This project uses CMake to handle linking LibTorch and OpenCV. From the root of the project, run:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

After compiling, a `bin/dcgan` binary is created. It takes a command-line argument to choose between training and inference. To train the DCGAN, use:
```bash
./bin/dcgan train
```
This saves training samples and the model checkpoints in `checkpoints/` directory in the root. For inference:
```bash
./bin/dcgan inference
```
Which saves a `inference-sample.png` in `checkpoints/`.
