#ifndef DOWNSCALEEXECUTER_CUH
#define DOWNSCALEEXECUTER_CUH

#include "DeviceExecuter.cuh"
#include <opencv2/opencv.hpp>

namespace craftify::executers {
    class DownscaleExecuter : public DeviceExecuter<cv::Mat, cv::Mat>{
    public:
        explicit DownscaleExecuter(unsigned int scale_factor = 16u);

        std::optional<std::shared_ptr<cv::Mat>> execute(std::shared_ptr<cv::Mat> input) override;

    private:
        unsigned int m_scale_factor;
    };
}


#endif //DOWNSCALEEXECUTER_CUH
