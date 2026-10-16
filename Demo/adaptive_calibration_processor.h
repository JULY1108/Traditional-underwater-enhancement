#pragma once
#ifndef ADAPTIVE_CALIBRATION_PROCESSOR_H
#define ADAPTIVE_CALIBRATION_PROCESSOR_H

#include <string>

namespace cv { class Mat; }

class AdaptiveCalibrationEnhance
{
public:
    explicit AdaptiveCalibrationEnhance(const std::string& model_path);
    ~AdaptiveCalibrationEnhance();

    cv::Mat process(const cv::Mat& src_bgr);

private:
    class Impl;
    Impl* pImpl;
};

#endif
