#include <iostream>
#include <memory>
#include "executers/DownscaleExecuter.cuh"
#include "executers/CameraFetcher.h"
#include "executers/ColorCVT.h"
#include "executers/ImgShow.h"

constexpr int CAMERA_ID = 1;
constexpr unsigned int SCALE_FACTOR = 2u;
constexpr const char *WINDOW_NAME = "Downscale";


int main() {
    bool running = true;

    auto start_module = std::make_shared<craftify::executers::CameraFetcher>(CAMERA_ID);

    std::vector<std::shared_ptr<craftify::executers::DeviceExecuter<cv::Mat, cv::Mat>>> executers;
    executers.emplace_back(std::make_shared<craftify::executers::ColorCVT>(cv::COLOR_RGB2RGBA));
    executers.emplace_back(std::make_shared<craftify::executers::DownscaleExecuter>(SCALE_FACTOR));
    executers.emplace_back(std::make_shared<craftify::executers::ImgShow>(WINDOW_NAME, running));

    while(running) {
        std::shared_ptr<cv::Mat> work_item = start_module->execute(nullptr);
        
        for (auto &executer : executers) {
            if (work_item == nullptr) {
                break;
            }
            work_item = executer->execute(work_item);
        }
    }
    return 0;
}
