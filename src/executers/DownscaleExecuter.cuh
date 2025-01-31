#ifndef DOWNSCALEEXECUTER_CUH
#define DOWNSCALEEXECUTER_CUH

#include "../pipeline_tools/Module.hpp"
#include <opencv2/opencv.hpp>

namespace craftify {
    namespace executers {
        class DownscaleExecuter : public pipeline_tools::Module<cv::Mat, cv::Mat>{
        public:
            explicit DownscaleExecuter(unsigned int scale_factor = 16u);

            std::shared_ptr<cv::Mat> process(std::shared_ptr<cv::Mat> input) override;

        private:
            unsigned int m_scale_factor;
        };
    }; // namespace executers
}; // namespace craftify

#endif //DOWNSCALEEXECUTER_CUH
