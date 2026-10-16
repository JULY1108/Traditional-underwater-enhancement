// 冒烟测试：合成一张模拟水下偏色 + 噪声图，跑完整 process()（透水 + 逐通道 CLAHE 线程结构），保存输出
#include <opencv2/opencv.hpp>
#include "dehaze_common.h"
#include <chrono>
#include <iostream>

int main()
{
    cv::Mat img(480, 640, CV_8UC3);
    cv::RNG rng(42);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            double x = (double)j / img.cols, y = (double)i / img.rows;
            float b = 130.f + 60.f * x + 25.f * y; // 蓝偏多，模拟水下
            float g = 70.f + 40.f * x + 15.f * y;
            float r = 45.f + 30.f * x + 10.f * y;
            b += (float)rng.gaussian(8);
            g += (float)rng.gaussian(8);
            r += (float)rng.gaussian(8);
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(cv::saturate_cast<uchar>(b),
                                                cv::saturate_cast<uchar>(g),
                                                cv::saturate_cast<uchar>(r));
        }
    }
    cv::imwrite("input_smoke.png", img);

    Rank1PlusUnderwater dehazer;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat out = dehazer.process(img, 1.0);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "process time: " << ms << " ms" << std::endl;
    std::cout << "out size: " << out.cols << "x" << out.rows
              << "  type: " << out.type() << "  empty: " << (out.empty() ? "yes" : "no") << std::endl;

    if (out.empty() || out.size() != img.size() || out.type() != CV_8UC3) return 1;
    if (!out.empty())
    {
        double mn, mx;
        cv::minMaxLoc(out, &mn, &mx);
        std::cout << "out range: [" << mn << ", " << mx << "]" << std::endl;
        if (cv::imwrite("output_smoke.png", out))
            std::cout << "saved output_smoke.png" << std::endl;
        else
            std::cout << "FAILED to save output" << std::endl;
    }
    return 0;
}

