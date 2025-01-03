#ifndef COLOR_CVT_H
#define COLOR_CVT_H

#include <opencv2/opencv.hpp>
#include "DeviceExecuter.cuh"

namespace craftify {
    namespace executers {
        class ColorCVT: public DeviceExecuter<cv::Mat, cv::Mat> {
            public:
                ColorCVT(int cvt_mode);

                std::shared_ptr<cv::Mat> execute(std::shared_ptr<cv::Mat> input) override;

            private:
                int m_cvt_mode;
            };
    }; // namespace executers
}; // namespace craftify

#endif //COLOR_CVT_H