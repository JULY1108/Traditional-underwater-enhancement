#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <opencv2/opencv.hpp>
#include "dehaze_common.h"

#include <omp.h>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "smooth_clahe.h"


class Rank1PlusUnderwater::Impl
{
public:
    Impl();
    ~Impl() = default;

    cv::Mat runProcess(cv::Mat input_img, double resize_scale);

    void Rank1_SnuT_iter(const cv::Mat& img, cv::Mat& T, cv::Mat& Snu);
    void get_atmosphere_rank1(const cv::Mat& img, const cv::Mat& tilde_T, cv::Mat& atmosphere);
    void fastTransRefine(cv::Mat& T_gray, const cv::Mat& guide, double lambda, double sigma);
    void dehaze(const cv::Mat& I, const cv::Mat& A, const cv::Mat& t, cv::Mat& J);

    struct MyParam
    {
        double omega = 0.85;
        double refine_lambda = 600;
        double refine_sigma = 3;
    } m_param;

    void guidedFilterSingle(const cv::Mat& guide, const cv::Mat& src, cv::Mat& dst, int r, float eps);
};

Rank1PlusUnderwater::Impl::Impl() = default;





// 通道耦合 + 高光/暗部保护的百分比拉伸（浑水场景参数化表）：
// - 三通道合并统计直方图，lo/hi 分位数共用一套阈值（每个像素的通道间比例不变 → 消除逐通道拉伸的色偏）
// - protectHighlights=true：hi 以上的像素视为回散射高光区，线性软过渡到 255，不截断；
//   false：退化为纯百分比拉伸，高光直接截断到 255
// - protectShadows=true：浑水暗部是鱼/水底细节而非噪声，lo 映射到 lo_out(15) 而非 0，
//   lo 以下 [0, lo] 线性软过渡，保全暗部层次不被压死；false：低分位以下直接压 0
static void percentileStretch(const cv::Mat& src, cv::Mat& dst,
                              double lo_percent = 1.0, double hi_percent = 99.0,
                              bool protectHighlights = true, bool protectShadows = true)
{
    CV_Assert(src.type() == CV_8UC3);
    const int N = src.rows * src.cols;

    // 三通道合并直方图（共用一套阈值的关键：单条 LUT 作用于全部通道）
    int hist[256] = { 0 };
    for (int i = 0; i < N; ++i)
    {
        const cv::Vec3b& v = src.at<cv::Vec3b>(i);
        hist[v[0]]++;
        hist[v[1]]++;
        hist[v[2]]++;
    }

    // 分位数阈值：lo 取累计 ≤ lo_percent 的最后一个 bin，hi 取累计 ≥ hi_percent 的第一个 bin
    const long total = (long)N * 3;
    const long targetLo = (long)(total * lo_percent / 100.0);
    const long targetHi = (long)(total * hi_percent / 100.0);
    int lo = 0, hi = 255;
    long cum = 0;
    for (int b = 0; b < 256; ++b)
    {
        cum += hist[b];
        if (cum <= targetLo) lo = b;
        if (cum >= targetHi) { hi = b; break; }
    }
    if (hi <= lo) { dst = src.clone(); return; }   // 退化情况（分布无跨度），原样返回

    // 高光保护时 hi 映射到 hi_out 而非 255，留出 5 级灰度给高光区软过渡，保证曲线单调不回折
    const double hi_out = 250.0;
    const double lo_out = 15.0;                                  // 暗部保护：lo 映射到 15 而非 0
    const double scale = (hi_out - lo_out) / (double)(hi - lo);
    uchar lutArr[256];
    if (protectShadows && lo > 0)
    {
        // 暗部软保护：v ∈ [0, lo] 线性 0→lo_out，纯保暗部层次；与拉伸段在 lo 处连续
        for (int v = 0; v <= lo; ++v) lutArr[v] = (uchar)(lo_out * v / (double)lo + 0.5);
    }
    else
    {
        for (int v = 0; v <= lo; ++v) lutArr[v] = 0;                              // 低分位以下压到 0
    }
    for (int v = lo + 1; v <= hi; ++v) lutArr[v] = (uchar)(lo_out + (v - lo) * scale + 0.5); // lo~hi 线性拉伸
    if (protectHighlights && hi < 255)
    {
        const double ramp = (255.0 - hi_out) / (255 - hi);                        // 高光区线性过渡到 255
        for (int v = hi + 1; v <= 255; ++v) lutArr[v] = (uchar)(hi_out + (v - hi) * ramp + 0.5);
    }
    else
    {
        for (int v = hi + 1; v <= 255; ++v) lutArr[v] = 255;                      // 纯百分比拉伸：高光截断
    }

    cv::Mat lut(1, 256, CV_8UC1, lutArr);
    cv::LUT(src, lut, dst);
}

// 逐通道百分比拉伸：每个通道独立统计直方图、独立求 lo/hi 分位数、独立建 LUT，
// 各通道动态范围被独立均衡——衰减最严重的通道（如水下红通道）会被单独放大，
// 可明显纠正蓝绿色偏。
// 注意：与 percentileStretch 不同，各通道拉伸比例不同，若某通道直方图分布异常
// 可能引入新的色偏——需要保色相时请切回合并直方图版本（percentileStretch）。
static void percentileStretchPerChannel(const cv::Mat& src, cv::Mat& dst,
                                        double lo_percent = 1.0, double hi_percent = 99.0,
                                        bool protectHighlights = true, bool protectShadows = true)
{
    CV_Assert(src.type() == CV_8UC3);
    const int N = src.rows * src.cols;

    const double hi_out = 250.0;
    const double lo_out = 1.0;
    uchar lut[3][256];
    int hist[3][256] = { { 0 } };

    // 单趟遍历，同时统计三个通道的直方图
    for (int i = 0; i < N; ++i)
    {
        const cv::Vec3b& v = src.at<cv::Vec3b>(i);
        hist[0][v[0]]++;
        hist[1][v[1]]++;
        hist[2][v[2]]++;
    }

    for (int ch = 0; ch < 3; ++ch)
    {
        // 分位数阈值：lo 取累计 ≤ lo_percent 的最后一个 bin，hi 取累计 ≥ hi_percent 的第一个 bin
        const long total = (long)N;
        const long targetLo = (long)(total * lo_percent / 100.0);
        const long targetHi = (long)(total * hi_percent / 100.0);
        int lo = 0, hi = 255;
        long cum = 0;
        for (int b = 0; b < 256; ++b)
        {
            cum += hist[ch][b];
            if (cum <= targetLo) lo = b;
            if (cum >= targetHi) { hi = b; break; }
        }

        if (hi <= lo)   // 该通道分布无跨度：退化为恒等映射
        {
            for (int v = 0; v < 256; ++v) lut[ch][v] = (uchar)v;
            continue;
        }

        const double scale = (hi_out - lo_out) / (double)(hi - lo);
        if (protectShadows && lo > 0)
        {
            // 暗部软保护：v ∈ [0, lo] 线性 0→lo_out，保住暗部层次；与拉伸段在 lo 处连续
            for (int v = 0; v <= lo; ++v) lut[ch][v] = (uchar)(lo_out * v / (double)lo + 0.5);
        }
        else
        {
            for (int v = 0; v <= lo; ++v) lut[ch][v] = 0;   // 低分位以下压到 0
        }
        for (int v = lo + 1; v <= hi; ++v) lut[ch][v] = (uchar)(lo_out + (v - lo) * scale + 0.5); // lo~hi 线性拉伸
        if (protectHighlights && hi < 255)
        {
            const double ramp = (255.0 - hi_out) / (255 - hi);   // 高光区线性过渡到 255
            for (int v = hi + 1; v <= 255; ++v) lut[ch][v] = (uchar)(hi_out + (v - hi) * ramp + 0.5);
        }
        else
        {
            for (int v = hi + 1; v <= 255; ++v) lut[ch][v] = 255;   // 纯百分比拉伸：高光截断
        }
    }

    // 三通道各自应用 LUT 后合并
    std::vector<cv::Mat> chans(3);
    cv::split(src, chans);
    for (int c = 0; c < 3; ++c)
    {
        cv::Mat l(1, 256, CV_8UC1, lut[c]);
        cv::LUT(chans[c], l, chans[c]);
    }
    cv::merge(chans, dst);
}



cv::Mat Rank1PlusUnderwater::Impl::runProcess(cv::Mat input_img, double resize_scale)
{
    int w = cvRound(input_img.cols * resize_scale);
    int h = cvRound(input_img.rows * resize_scale);
    cv::resize(input_img, input_img, cv::Size(w, h));

    // Physical dehazing is disabled here; start from the original 8-bit image.
    cv::Mat J;
    // Pipeline stage owned by this class: SmoothCLAHE -> percentile stretch.
    // Adaptive calibration is invoked by demo_TSA.cpp before this post-process stage.
    J = input_img;
    // Optional superpixel/WLS dehazing can replace J with a float [0, 1] result.
    // The channel stage converts that result back to 8-bit automatically.
    // See superpixelWLSPipeline for its tuning parameters.




    // ---------- 后处理：逐通道 CLAHE（对齐 Demo_非偏透水_2 的两阶段线程结构） ----------
    // 分配 6 张 float 单通道图（mIr/mIg/mIb 存 CLAHE 后的 float 通道，mJr/mJg/mJb 存归一化结果）+ 1 张三通道图 mJ
    cv::Mat mIr, mIg, mIb, mJr, mJg, mJb;
    std::vector<cv::Mat*> iMats = { &mIr, &mIg, &mIb };
    std::vector<cv::Mat*> jMats = { &mJr, &mJg, &mJb };
    for (cv::Mat* p : iMats) p->create(h, w, CV_32FC1);
    for (cv::Mat* p : jMats) p->create(h, w, CV_32FC1);

    // 计时（对应 _2 的 preciseStopwatch，默认注释掉）：
    // auto t_start = std::chrono::steady_clock::now();

    // 阶段1：3 线程并行 calculateI(0/1/2) —— 提取通道 → CLAHE → 转回 float（与 _2 的 calculateI 同构）。
    // 注意：每个线程必须持有自己的 SmoothCLAHE 实例——apply 会改写成员变量（gridX_/gridY_），实例不能跨线程共享。
    auto calculateI = [&](int c) {
        SmoothCLAHE clahe_enhanced(16, cv::Size(64, 64), true, true);
        cv::Mat chan;
        cv::extractChannel(J, chan, c);                           // Extract one channel from the original image.
        if (chan.type() != CV_8UC1)
            chan.convertTo(chan, CV_8UC1, 255.0);                 // Convert a float [0, 1] dehazing result to 8-bit.



        clahe_enhanced.apply(chan, chan);                         // 通道内原地 CLAHE（输出像素只依赖同位像素，原地安全）
        chan.convertTo(*iMats[c], CV_32F, 1.0f / 128.0f);                        // 转回 float（0~255）
    };
    std::vector<std::thread> t1;
    for (int c = 0; c < 3; ++c) t1.emplace_back(calculateI, c);
    for (auto& t : t1) t.join();
    // auto t_mid = std::chrono::steady_clock::now();  // 阶段1耗时 = duration<double, milli>(t_mid - t_start).count()

    // 阶段2：3 线程并行 ReconstructImage(0/1/2) —— 各通道按最大值归一化回 8U。
    // 注：CLAHE 的 LUT 已把最大灰度映射到 255，故此步实际近似恒等；保留是为了结构上与 _2 对齐。
    auto reconstruct = [&](int c) {
        double maxVal = 0.0;
        cv::minMaxLoc(*iMats[c], nullptr, &maxVal);
        float scale = (maxVal > 1e-6) ? (255.0f / (float)maxVal) : 0.0f;
        iMats[c]->convertTo(*jMats[c], CV_8UC1, scale);           // J = I/max*255，饱和截断
    };
    std::vector<std::thread> t2;
    for (int c = 0; c < 3; ++c) t2.emplace_back(reconstruct, c);
    for (auto& t : t2) t.join();
    // auto t_end = std::chrono::steady_clock::now();  // 阶段2耗时 = duration<double, milli>(t_end - t_mid).count()

    // merge 三通道 → BGR 三通道图 mJ
    cv::Mat mJ;
    cv::merge(std::vector<cv::Mat>{mJr, mJg, mJb}, mJ); 

    // ---------- 对比度后处理：百分比拉伸（浑水参数） ----------
    percentileStretch(mJ, mJ, 1.0, 99.0, true, true);
   //percentileStretchPerChannel(mJ, mJ, 1.0, 99.0, true, true);

    return mJ;
}

Rank1PlusUnderwater::Rank1PlusUnderwater()
    : pImpl(new Impl())
{}

Rank1PlusUnderwater::~Rank1PlusUnderwater()
{
    delete pImpl;
}

cv::Mat Rank1PlusUnderwater::process(cv::Mat input_img, double resize_scale)
{
    return pImpl->runProcess(std::move(input_img), resize_scale);
}
