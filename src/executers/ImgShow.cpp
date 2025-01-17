#include "ImgShow.h"

namespace craftify {
    namespace executers {
        ImgShow::ImgShow(const std::string &win_title, bool &running_ref): m_win_title(win_title), m_running_ref(running_ref) {}

        std::shared_ptr<cv::Mat>
        ImgShow::process(std::shared_ptr<cv::Mat> input) {
            cv::imshow(m_win_title, *input.get());
            if (cv::waitKey(std::floor(1000 / 60)) == 'q') {
                m_running_ref = false;
            }

            return nullptr;
        }
    }; // namespace executers
}; // namespace craftify