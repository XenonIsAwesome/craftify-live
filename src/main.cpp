#include <iostream>
#include <memory>
#include "executers/DownscaleExecuter.cuh"
#include "executers/CameraFetcher.h"
#include "executers/ColorCVT.h"
#include "executers/ImgShow.h"
#include "executers/BlockifyExecuter.cuh"
#include "pipeline_tools/UntypedModule.h"
#include "pipeline_tools/Pipeline.h"

constexpr int CAMERA_ID = 1;
constexpr unsigned int PRE_BLOCKIFY_DOWNSCALE_FACTOR = 4u;
constexpr unsigned int POST_BLOCKIFY_DOWNSCALE_FACTOR = 1u;

constexpr const char *WINDOW_NAME = "Blockify";
constexpr const char *ATLAS_PATH = "assets/atlas.png";
constexpr const char *AVG_COLORS_PATH = "assets/colors.bin";

int main() {
    bool running = true;

    std::vector<std::shared_ptr<pipeline_tools::UntypedModule>> modules;
     modules.emplace_back(std::make_shared<craftify::executers::CameraFetcher>(CAMERA_ID));
    modules.emplace_back(std::make_shared<craftify::executers::ColorCVT>(cv::COLOR_BGR2RGBA));
    modules.emplace_back(std::make_shared<craftify::executers::DownscaleExecuter>(PRE_BLOCKIFY_DOWNSCALE_FACTOR));
    modules.emplace_back(std::make_shared<craftify::executers::BlockifyExecuter>(ATLAS_PATH, AVG_COLORS_PATH));
    modules.emplace_back(std::make_shared<craftify::executers::DownscaleExecuter>(POST_BLOCKIFY_DOWNSCALE_FACTOR));  // TODO: Switch with FitToScreen
    modules.emplace_back(std::make_shared<craftify::executers::ImgShow>(WINDOW_NAME, running));

    auto pipeline = std::make_shared<pipeline_tools::Pipeline>(modules);
    pipeline->start();

    while(running);

    pipeline->stop();
}
