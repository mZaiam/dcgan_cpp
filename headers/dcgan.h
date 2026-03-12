#pragma once
#include <torch/torch.h>

namespace nn = torch::nn;

// Generator
struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kLatentDim);
    torch::Tensor forward(torch::Tensor x);

    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);

// Discriminator
struct DCGANDiscriminatorImpl : nn::Module {
    DCGANDiscriminatorImpl(int kInputChannels);
    torch::Tensor forward(torch::Tensor x);

    nn::Conv2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANDiscriminator);