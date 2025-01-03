#include "DownscaleExecuter.cuh"
#include "../kernels/downscale.cuh"

namespace craftify::executers {
    std::optional<std::shared_ptr<cv::Mat>>
    DownscaleExecuter::execute(std::shared_ptr<cv::Mat> input) {
        unsigned int inputWidth = input->cols;
        unsigned int inputHeight = input->rows;
        unsigned int outputWidth = inputWidth / m_scale_factor;
        unsigned int outputHeight = inputHeight / m_scale_factor;

        auto output = std::make_shared<cv::Mat>(outputHeight, outputWidth, input->type());

        uchar4 *d_input, *d_output;
        size_t inputSize = inputWidth * inputHeight * sizeof(uchar4);
        size_t outputSize = outputWidth * outputHeight * sizeof(uchar4);

        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);

        cudaMemcpy(d_input, input->data, inputSize, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

        craftify::kernels::downscale_image<<<gridSize, blockSize>>>(
            d_input, d_output, 
            inputWidth, inputHeight, 
            outputWidth, outputHeight, 
            m_scale_factor);
        
        cudaMemcpy(output->data, d_output, outputSize, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        cudaDeviceSynchronize();

        return output;
    }

    DownscaleExecuter::DownscaleExecuter(unsigned int scale_factor): m_scale_factor(scale_factor) {}
};