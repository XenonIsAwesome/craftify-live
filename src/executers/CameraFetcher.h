#ifndef CAMERA_FETCHER_H
#define CAMERA_FETCHER_H

#include <opencv2/opencv.hpp>
#include "DeviceExecuter.cuh"

namespace craftify {
    namespace executers {
        class CameraFetcher: public DeviceExecuter<void, cv::Mat> {
        public:
            CameraFetcher(int cam_id = 0);
            ~CameraFetcher();

            std::shared_ptr<cv::Mat> execute(std::shared_ptr<void> input) override;

        private:
            cv::VideoCapture cap;
        };
    }; // namespace executers
}; // namespace craftify

#endif //CAMERA_FETCHER_H