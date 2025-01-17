#ifndef IMGSHOW_H
#define IMGSHOW_H

#include <opencv2/opencv.hpp>
#include "../pipeline_tools/Module.hpp"
#include <string>

namespace craftify {
    namespace executers {
        class ImgShow: public pipeline_tools::Module<cv::Mat, cv::Mat> {
            public:
                ImgShow(const std::string &win_title, bool &running_ref);

                std::shared_ptr<cv::Mat> process(std::shared_ptr<cv::Mat> input) override;

            private:
                std::string m_win_title;
                bool &m_running_ref;
            };
    }; // namespace executers
}; // namespace craftify

#endif //IMGSHOW_H