#include "ColorCVT.h"

namespace craftify {
    namespace executers {
        ColorCVT::ColorCVT(int cvt_mode): m_cvt_mode(cvt_mode) {}

        std::shared_ptr<cv::Mat>
        ColorCVT::process(std::shared_ptr<cv::Mat> input) {
            auto output = std::make_shared<cv::Mat>();
            cv::cvtColor(*input, *output, m_cvt_mode);
            // cv::resize(*output, *output, cv::Size(input->cols / 16, input->rows / 16), 0, 0, cv::INTER_LANCZOS4);

            return output;
        }
    }; // namespace executers
}; // namespace craftify