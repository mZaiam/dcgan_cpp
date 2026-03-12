#include "dcgan.h"

// Generator
DCGANGeneratorImpl::DCGANGeneratorImpl(int kLatentDim)
    : conv1(nn::ConvTranspose2dOptions(kLatentDim, 256, 4).bias(false)),
      batch_norm1(256),
      conv2(nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
      batch_norm2(128),
      conv3(nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
      batch_norm3(64),
      conv4(nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false)) 
{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("batch_norm1", batch_norm1);
    register_module("batch_norm2", batch_norm2);
    register_module("batch_norm3", batch_norm3);
}

torch::Tensor DCGANGeneratorImpl::forward(torch::Tensor x) {
    x = torch::relu(batch_norm1(conv1(x)));
    x = torch::relu(batch_norm2(conv2(x)));
    x = torch::relu(batch_norm3(conv3(x)));
    x = torch::tanh(conv4(x));
    return x;
}

// Discriminator
DCGANDiscriminatorImpl::DCGANDiscriminatorImpl(int kInputChannels)
    : conv1(nn::Conv2dOptions(kInputChannels, 64, 4).stride(2).padding(1).bias(false)),
      conv2(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      batch_norm2(128),
      conv3(nn::Conv2dOptions(128, 256, 3).stride(2).padding(1).bias(false)),
      batch_norm3(256),
      conv4(nn::Conv2dOptions(256, 1, 4).stride(1).padding(0).bias(false)) 
{
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("batch_norm2", batch_norm2);
    register_module("conv3", conv3);
    register_module("batch_norm3", batch_norm3);
    register_module("conv4", conv4);
}

torch::Tensor DCGANDiscriminatorImpl::forward(torch::Tensor x) {
    x = torch::leaky_relu(conv1(x), 0.2);        
    x = torch::leaky_relu(batch_norm2(conv2(x)), 0.2);
    x = torch::leaky_relu(batch_norm3(conv3(x)), 0.2);
    x = torch::sigmoid(conv4(x));        
    return x;
}