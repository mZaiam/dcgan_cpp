#include <iostream>
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp> 

#include "dcgan.h"
#include "train.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: choose a mode.\n";
        return 1;
    }

    std::string mode = argv[1];

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    constexpr int64_t kLatentDim = 100;
    DCGANGenerator generator(kLatentDim);
    generator->to(device);

    std::string checkpointDir = std::string(PROJECT_SOURCE_DIR) + "/checkpoints/";

    if (mode == "train") {
        std::cout << "--- Train ---\n";
        
        constexpr int64_t kBatchSize = 64;
        constexpr int64_t kEpochs = 10;
        constexpr bool kRestoreFromCheckpoint = false;

        DCGANDiscriminator discriminator(1);
        discriminator->to(device);

        std::string dataPath = std::string(PROJECT_SOURCE_DIR) + "/data";
        
        train_dcgan(
            generator, 
            discriminator, 
            dataPath, 
            kEpochs, 
            kBatchSize, 
            kLatentDim, 
            kRestoreFromCheckpoint, 
            device
        );

        std::cout << "Train done." << std::endl;
    } else if (mode == "inference") {
        std::cout << "--- Inference ---\n";
        
        try {
            torch::load(generator, checkpointDir + "generator-checkpoint.pt");
            std::cout << "Weights loaded.\n";
        } catch (...) {
            std::cerr << "Failed to load weights from " << checkpointDir << "\n";
            return 1;
        }
    
        generator->eval(); 
        torch::NoGradGuard noGrad; 

        auto noise = torch::randn({1, kLatentDim, 1, 1}, device);
        auto output = generator->forward(noise);

        output = (output + 1.0) / 2.0;

        output = output.mul(255).clamp(0, 255).to(torch::kU8);        
        output = output.squeeze();         
        output = output.to(torch::kCPU).contiguous();

        cv::Mat img(28, 28, CV_8UC1, output.data_ptr());
        std::string outPath = checkpointDir + "inference-sample.png";
        cv::imwrite(outPath, img);
        
        std::cout << "Inference done." << std::endl;
        
    } else {
        std::cerr << "Unknown mode.\n";
        return 1;
    } 

    return 0;
}