#include "BlockifyExecuter.cuh"
#include "../kernels/blockify.cuh"
#include <fstream>

constexpr size_t BLOCK_SIZE = 16;

namespace craftify {
    namespace executers {

        BlockifyExecuter::BlockifyExecuter(const std::string &texture_atlas_path, const std::string &avg_colors_path) {
            auto texture_atlas = load_texture_atlas(texture_atlas_path);
            auto avg_colors = load_avg_colors(avg_colors_path);

            atlas_width = texture_atlas.cols;
            atlas_height = texture_atlas.rows;
            avg_colors_size = avg_colors.size();


            size_t atlas_size = atlas_width * atlas_height * sizeof(uchar4);
            size_t avg_colors_size = avg_colors.size() * sizeof(unsigned long);

            cudaMalloc(&d_atlas, atlas_size);
            cudaMalloc(&d_avg_colors, avg_colors_size);

            cudaMemcpy(d_atlas, texture_atlas.data, atlas_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_avg_colors, avg_colors.data(), avg_colors_size, cudaMemcpyHostToDevice);

            texture_atlas.release();
        }

        BlockifyExecuter::~BlockifyExecuter() {
            cudaFree(d_atlas);
            cudaFree(d_avg_colors);
        }

        cv::Mat BlockifyExecuter::load_texture_atlas(const std::string &texture_atlas_path) {
            cv::Mat texture_atlas = cv::imread(texture_atlas_path, cv::IMREAD_UNCHANGED);
            cv::cvtColor(texture_atlas, texture_atlas, cv::COLOR_RGB2RGBA);

            return texture_atlas;
        }

        std::vector<size_t> BlockifyExecuter::load_scores(const std::string &scores_path) {
            std::ifstream file(scores_path, std::ios::binary);

            if (file) {
                file.seekg(0, std::ios::end);
                size_t file_size = file.tellg();
                file.seekg(0, std::ios::beg);

                size_t num_elements = file_size / sizeof(unsigned long);
                std::vector<size_t> buffer(num_elements);

                file.read(reinterpret_cast<char*>(buffer.data()), file_size);

                return buffer;
            } else {
                throw std::runtime_error("Could not open file: " + scores_path);
            }
        }


        std::vector<uchar4> BlockifyExecuter::load_avg_colors(const std::string &avg_colors_path) {
            std::ifstream file(avg_colors_path, std::ios::binary);

            if (file) {
                file.seekg(0, std::ios::end);
                size_t file_size = file.tellg();
                file.seekg(0, std::ios::beg);

                size_t num_elements = file_size / sizeof(uchar4);
                std::vector<uchar4> buffer(num_elements);

                file.read(reinterpret_cast<char*>(buffer.data()), file_size);

                return buffer;
            } else {
                throw std::runtime_error("Could not open file: " + avg_colors_path);
            }
        }

        std::shared_ptr<cv::Mat>
        BlockifyExecuter::process(std::shared_ptr<cv::Mat> input) {
            unsigned int inputWidth = input->cols;
            unsigned int inputHeight = input->rows;
            unsigned int outputWidth = inputWidth * BLOCK_SIZE;
            unsigned int outputHeight = inputHeight * BLOCK_SIZE;

            auto output = std::make_shared<cv::Mat>(outputHeight, outputWidth, CV_8UC4);

            uchar4 *d_input, *d_output;
            size_t inputSize = inputWidth * inputHeight * sizeof(uchar4);
            size_t outputSize = outputWidth * outputHeight * sizeof(uchar4);

            cudaMalloc(&d_input, inputSize);
            cudaMalloc(&d_output, outputSize);

            cudaMemcpy(d_input, input->data, inputSize, cudaMemcpyHostToDevice);

            dim3 blockSize(16, 16);
            dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

            craftify::kernels::blockify_image<<<gridSize, blockSize>>>(
                d_input, d_output, 
                d_atlas, d_avg_colors, avg_colors_size,

                inputWidth, inputHeight, 
                outputWidth, outputHeight, 
                atlas_width, atlas_height);
            
            cudaMemcpy(output->data, d_output, outputSize, cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_output);

            cudaDeviceSynchronize();

            return output;
        }

    }; // namespace executers
}; // namespace craftify