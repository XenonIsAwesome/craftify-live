#include "CameraFetcher.h"

namespace craftify {
    namespace executers {
        CameraFetcher::CameraFetcher(int cam_id) {
            cap = cv::VideoCapture(cam_id);

            // Check if the camera opened successfully.
            if (!cap.isOpened()) {
                std::stringstream err;
                err << "Error: Could not open the camera.";

                std::cerr << err.str() << std::endl;
                throw std::runtime_error(err.str());
            }
        }

        CameraFetcher::~CameraFetcher() {
            cap.release();
        }

        std::shared_ptr<cv::Mat>
        CameraFetcher::process(std::shared_ptr<void> input) {
            auto frame = std::make_shared<cv::Mat>();
            cap >> *frame.get();

            return frame;
        }
    }; // namespace executers
}; // namespace craftify