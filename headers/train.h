#pragma once
#include <string>
#include <torch/torch.h>
#include "dcgan.h"

void train_dcgan(
    DCGANGenerator& generator,
    DCGANDiscriminator& discriminator,
    const std::string& dataPath,
    int kNumEpochs,
    int kBatchSize,
    int kLatendDim,
    bool kRestoreFromCheckpoint,
    torch::Device device
);