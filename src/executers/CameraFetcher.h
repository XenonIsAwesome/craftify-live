#ifndef CAMERA_FETCHER_H
#define CAMERA_FETCHER_H

#include <opencv2/opencv.hpp>
#include "../pipeline_tools/Module.hpp"

namespace craftify {
    namespace executers {
        class CameraFetcher: public pipeline_tools::Module<void, cv::Mat> {
        public:
            CameraFetcher(int cam_id = 0);
            ~CameraFetcher();

            std::shared_ptr<cv::Mat> process(std::shared_ptr<void> input) override;

        private:
            cv::VideoCapture cap;
        };
    }; // namespace executers
}; // namespace craftify

#endif //CAMERA_FETCHER_H