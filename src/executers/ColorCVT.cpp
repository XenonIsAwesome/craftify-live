#include "ColorCVT.h"

namespace craftify {
    namespace executers {
        ColorCVT::ColorCVT(int cvt_mode): m_cvt_mode(cvt_mode) {}

        std::shared_ptr<cv::Mat>
        ColorCVT::execute(std::shared_ptr<cv::Mat> input) {
            auto output = std::make_shared<cv::Mat>();
            cv::cvtColor(*input, *output, m_cvt_mode);

            return output;
        }
    }; // namespace executers
}; // namespace craftify