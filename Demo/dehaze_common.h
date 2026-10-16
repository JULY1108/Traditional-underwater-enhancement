#pragma once
#ifndef DEHAZE_COMMON_H
#define DEHAZE_COMMON_H

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 仅保留导出类，彻底删除全局C接口
EXPORT class Rank1PlusUnderwater
{
public:
    Rank1PlusUnderwater();
    ~Rank1PlusUnderwater();
    cv::Mat process(cv::Mat input_img, double resize_scale);

private:
    class Impl;
    Impl* pImpl;
};

#endif