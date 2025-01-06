#include <iostream>
#include <memory>
#include "executers/DownscaleExecuter.cuh"
#include "executers/CameraFetcher.h"
#include "executers/ColorCVT.h"
#include "executers/ImgShow.h"
#include "executers/BlockifyExecuter.cuh"

constexpr int CAMERA_ID = 1;
constexpr unsigned int PRE_BLOCKIFY_DOWNSCALE_FACTOR = 4u;
constexpr unsigned int POST_BLOCKIFY_DOWNSCALE_FACTOR = 1u;

constexpr const char *WINDOW_NAME = "Blockify";
constexpr const char *ATLAS_PATH = "assets/atlas.png";
constexpr const char *AVG_COLORS_PATH = "assets/colors.bin";

int main() {
    bool running = true;

    auto start_module = std::make_shared<craftify::executers::CameraFetcher>(CAMERA_ID);

    std::vector<std::shared_ptr<craftify::executers::DeviceExecuter<cv::Mat, cv::Mat>>> executers;
    executers.emplace_back(std::make_shared<craftify::executers::ColorCVT>(cv::COLOR_BGR2RGBA));
    executers.emplace_back(std::make_shared<craftify::executers::DownscaleExecuter>(PRE_BLOCKIFY_DOWNSCALE_FACTOR));
    executers.emplace_back(std::make_shared<craftify::executers::BlockifyExecuter>(ATLAS_PATH, AVG_COLORS_PATH));
    executers.emplace_back(std::make_shared<craftify::executers::DownscaleExecuter>(POST_BLOCKIFY_DOWNSCALE_FACTOR));
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
