#ifndef FITTOSCREENEXECUTER_CUH
#define FITTOSCREENEXECUTER_CUH

#include "pipeline_tools/Module.hpp"
#include <opencv2/opencv.hpp>

namespace craftify {
    namespace executers {
        class FitToScreenExecuter : public pipeline_tools::Module<cv::Mat, cv::Mat>{
        public:
            explicit FitToScreenExecuter(unsigned int screen_width, unsigned int screen_height);

            std::shared_ptr<cv::Mat> process(std::shared_ptr<cv::Mat> input) override;

        private:
            unsigned int m_scr_width;
            unsigned int m_scr_height;
        };
    }; // namespace executers
}; // namespace craftify

#endif //FITTOSCREENEXECUTER_CUH
