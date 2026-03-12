#include "train.h"
#include <iostream>
#include <opencv2/opencv.hpp> 

void train_dcgan(
    DCGANGenerator& generator,
    DCGANDiscriminator& discriminator,
    const std::string& dataPath,
    int kNumEpochs,
    int kBatchSize,
    int kLatentDim,
    bool kRestoreFromCheckpoint,
    torch::Device device) 
{
    auto dataset = torch::data::datasets::MNIST(dataPath)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    const size_t batchesPerEpoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

    torch::optim::Adam generatorOptimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4)
                                       .betas(std::make_tuple(0.5, 0.999)));
    torch::optim::Adam discriminatorOptimizer(
        discriminator->parameters(), torch::optim::AdamOptions(2e-4)
                                           .betas(std::make_tuple(0.5, 0.999)));

    int64_t checkpointCounter = 0;
    std::string checkpointDir = std::string(PROJECT_SOURCE_DIR) + "/checkpoints/";

    if (kRestoreFromCheckpoint) {
        std::cout << "Restoring checkpoints from: " << checkpointDir << std::endl;

        torch::load(generator, checkpointDir + "generator-checkpoint.pt");
        torch::load(generatorOptimizer, checkpointDir + "generator-optimizer-checkpoint.pt");
        torch::load(discriminator, checkpointDir + "discriminator-checkpoint.pt");
        torch::load(discriminatorOptimizer, checkpointDir + "discriminator-optimizer-checkpoint.pt");
    }

    int kCheckpointEvery = 1; 

    for (int64_t epoch = 1; epoch <= kNumEpochs; ++epoch) {
        int64_t batchIndex = 0;
        
        for (torch::data::Example<>& batch : *dataLoader) {
            
            torch::Tensor realImages = batch.data.to(device);
            int64_t currentBatchSize = realImages.size(0);

            // Discriminator
            discriminator->zero_grad();

            torch::Tensor realLabels = torch::empty(currentBatchSize, device).uniform_(0.8, 1.0);
            torch::Tensor realOutput = discriminator->forward(realImages).reshape(realLabels.sizes());
            torch::Tensor dLossReal = torch::binary_cross_entropy(realOutput, realLabels);
            dLossReal.backward();

            torch::Tensor noise = torch::randn({currentBatchSize, kLatentDim, 1, 1}, device);
            torch::Tensor fakeImages = generator->forward(noise);
            torch::Tensor fakeLabels = torch::zeros(currentBatchSize, device);
            
            torch::Tensor fakeOutput = discriminator->forward(fakeImages.detach()).reshape(fakeLabels.sizes());
            torch::Tensor dLossFake = torch::binary_cross_entropy(fakeOutput, fakeLabels);
            dLossFake.backward();

            torch::Tensor dLoss = dLossReal + dLossFake;
            discriminatorOptimizer.step();

            // Generator
            generator->zero_grad();
            
            fakeLabels.fill_(1);
            fakeOutput = discriminator->forward(fakeImages).reshape(fakeLabels.sizes());
            torch::Tensor gLoss = torch::binary_cross_entropy(fakeOutput, fakeLabels);
            gLoss.backward();
            
            generatorOptimizer.step();

            std::printf(
                "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                epoch, kNumEpochs, ++batchIndex, batchesPerEpoch,
                dLoss.item<float>(), gLoss.item<float>());
            std::fflush(stdout);

            if (batchIndex % kCheckpointEvery == 0) {
                torch::save(generator, checkpointDir + "generator-checkpoint.pt");
                torch::save(generatorOptimizer, checkpointDir + "generator-optimizer-checkpoint.pt");
                torch::save(discriminator, checkpointDir + "discriminator-checkpoint.pt");
                torch::save(discriminatorOptimizer, checkpointDir + "discriminator-optimizer-checkpoint.pt");
                
                torch::Tensor samples = generator->forward(torch::randn({1, kLatentDim, 1, 1}, device));
                samples = (samples + 1.0) / 2.0; 
                
                samples = samples.mul(255).clamp(0, 255).to(torch::kU8);        
                samples = samples.squeeze();         
                samples = samples.to(torch::kCPU).contiguous();

                cv::Mat img(28, 28, CV_8UC1, samples.data_ptr());
                std::string outPath = checkpointDir + "train-sample" + std::to_string(epoch) + ".png";
                cv::imwrite(outPath, img);             
            }
        }
    }
}
