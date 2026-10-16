#pragma once
#ifndef LUT_BATCH_PROCESSOR_H
#define LUT_BATCH_PROCESSOR_H

#include <string>
namespace cv { class Mat; }

#if defined(_WIN32) || defined(_WIN64)
#define EXPORT __declspec(dllexport)
#define OS_WINDOWS 1
#elif defined(__linux__)
#define EXPORT __attribute__((visibility("default")))
#define OS_LINUX 1
#else
#error "Unsupported platform"
#endif

// 对外唯一导出类，标准PIMPL，仅外壳
EXPORT class Lut3DEnhance
{
public:
    Lut3DEnhance(const std::string& model_path);
    ~Lut3DEnhance();

    cv::Mat process(const cv::Mat& src_bgr, cv::Size out_size);

private:
    class Impl;
    Impl* pImpl;
};

#endif