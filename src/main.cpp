//
// Created by ofek1 on 1/2/2025.
//

#include <iostream>
#include "executers/DownscaleExecuter.cuh"

int main() {
    // Open the default camera (usually the first connected camera).
    cv::VideoCapture cap(0); 

    // Check if the camera opened successfully.
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    cv::Mat frame;

    cap >> frame;
    cap.release();

    std::shared_ptr<cv::Mat> input = std::make_shared<cv::Mat>(frame);

    if (input->channels() < 4) {
        cv::cvtColor(frame, *input, cv::COLOR_RGB2RGBA);
    }

    if (input->empty()) {
        std::cerr << "Error: Could not open input image!" << std::endl;
        return -1;
    }

    craftify::executers::DownscaleExecuter downscaleExecuter(16u);
    auto result = downscaleExecuter.execute(input);

    if (result.has_value()) {
        cv::imwrite("output.jpg", *result.value().get());
    } else {
        std::cerr << "Error: Could not downscale image!" << std::endl;
        return -1;
    }

    return 0;
}
