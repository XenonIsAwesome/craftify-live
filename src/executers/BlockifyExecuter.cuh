#ifndef BLOCKIFY_EXECUTER_CUH
#define BLOCKIFY_EXECUTER_CUH

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector_types.h>
#include "DeviceExecuter.cuh"

namespace craftify {
    namespace executers {
        class BlockifyExecuter: public DeviceExecuter<cv::Mat, cv::Mat> {
            public:
                BlockifyExecuter(const std::string &texture_atlas_path, const std::string &avg_colors_path);
                ~BlockifyExecuter();

                std::shared_ptr<cv::Mat> execute(std::shared_ptr<cv::Mat> input) override;
                
            private:
                cv::Mat load_texture_atlas(const std::string &texture_atlas_path);
                std::vector<size_t> load_scores(const std::string &scores_path);
                std::vector<uchar4> load_avg_colors(const std::string &avg_colors_path);

                uchar4 *d_atlas;
                uchar4 *d_avg_colors;

                size_t atlas_width;
                size_t atlas_height;
                size_t avg_colors_size;
        };
    }; // namespace executers
}; // namespace craftify

#endif // BLOCKIFY_EXECUTER_CUH