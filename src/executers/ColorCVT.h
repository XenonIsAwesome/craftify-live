#ifndef COLOR_CVT_H
#define COLOR_CVT_H

#include <opencv2/opencv.hpp>
#include "../pipeline_tools/Module.hpp"

namespace craftify {
    namespace executers {
        class ColorCVT: public pipeline_tools::Module<cv::Mat, cv::Mat> {
            public:
                ColorCVT(int cvt_mode);

                std::shared_ptr<cv::Mat> process(std::shared_ptr<cv::Mat> input) override;

            private:
                int m_cvt_mode;
            };
    }; // namespace executers
}; // namespace craftify

#endif //COLOR_CVT_H