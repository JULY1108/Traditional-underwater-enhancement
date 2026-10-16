#ifndef SMOOTH_CLAHE_H
#define SMOOTH_CLAHE_H

#include <opencv2/opencv.hpp>
#include <vector>

class SmoothCLAHE {
public:
    // clipLimit: 对比度限制 (推荐 2.0 - 10.0)
    // tileGridSize: 分块大小 (推荐 16x16 或 32x32)
    // enableLUTFilter: 是否开启映射表空域滤波，去除块效应和图像四周的黑边
    SmoothCLAHE(double clipLimit = 4.0, cv::Size tileGridSize = cv::Size(16, 16), bool enableLUTFilter = true, bool useBilinear = true);

    // 调用自定义 CLAHE 的主函数
    void apply(const cv::Mat &src, cv::Mat &dst);

private:
    double clipLimit_;
    cv::Size tileSize_;
    bool filterEnable_;
    int gridX_, gridY_;
    bool useBilinear_;

    std::vector<int> ind1_;
    std::vector<int> ind2_;
    std::vector<float> xa_;
    std::vector<float> xa1_;

    // 核心逻辑：计算直方图、裁剪、生成 LUT
    void calculateMaps(const cv::Mat &src, std::vector<std::vector<cv::Mat>> &maps);
    void calculateMapsOptimized(const cv::Mat &src, std::vector<float> &lutBuffer);

    // 对 LUT 进行 3x3 加权平均，实现空域滤波
    void filterLUTs(std::vector<std::vector<cv::Mat>> &maps);
    void filterLUTsOptimized(std::vector<float> &lutBuffer);

};

#endif