#ifndef IMGSHOW_H
#define IMGSHOW_H

#include <opencv2/opencv.hpp>
#include "DeviceExecuter.cuh"
#include <string>

namespace craftify {
    namespace executers {
        class ImgShow: public DeviceExecuter<cv::Mat, cv::Mat> {
            public:
                ImgShow(const std::string &win_title, bool &running_ref);

                std::shared_ptr<cv::Mat> execute(std::shared_ptr<cv::Mat> input) override;

            private:
                std::string m_win_title;
                bool &m_running_ref;
            };
    }; // namespace executers
}; // namespace craftify

#endif //IMGSHOW_H