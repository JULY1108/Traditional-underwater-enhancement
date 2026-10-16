#include <opencv2/opencv.hpp>
#include "lut_batch_processor.h"
#include <torch/script.h>
#include <torch/cuda.h>

#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <unordered_map>
#include <memory>

// 全部内容收拢进Impl，无全局命名空间
class Lut3DEnhance::Impl
{
public:
    explicit Impl(const std::string& model_dir);
    ~Impl() = default;

    bool init();
    cv::Mat process(const cv::Mat& input_bgr, int out_w, int out_h);

    // LUT全局常量全部移入Impl作用域
    static constexpr int LUT_DIM = 33;
    static constexpr int LUT_SHIFT = LUT_DIM * LUT_DIM * LUT_DIM;
    static constexpr int LUT_SIZE = 3 * LUT_SHIFT;
    static constexpr float BINSIZE = 1.000001f / (LUT_DIM - 1);

    // 内部结构体
    struct TriLinearWS {
        torch::Tensor idx_buf;
        torch::Tensor w_buf;
        void ensure(int H, int W, const torch::Device& device) {
            int N = H * W;
            if (!idx_buf.defined() || idx_buf.size(1) != N) {
                auto li = torch::TensorOptions().dtype(torch::kLong).device(device);
                auto fi = torch::TensorOptions().dtype(torch::kFloat32).device(device);
                idx_buf = torch::empty({ 8, N }, li);
                w_buf = torch::empty({ 8, N }, fi);
            }
        }
    };

    // CPU三线性插值函数 成员化
    void TriLinearForwardCpu(const float* lut, const float* image, float* output,
        int dim, int shift, float binsize,
        int width, int height, int channels);

    // 私有成员变量
    std::string m_model_dir;
    torch::jit::script::Module m_classifier;
    torch::Device m_device = torch::Device(torch::kCPU);
    bool m_use_cuda = false;

    std::vector<float> m_lut0, m_lut1, m_lut2;
    std::vector<float> m_final_lut;

    torch::Tensor m_lut0_gpu, m_lut1_gpu, m_lut2_gpu;
    torch::Tensor m_lut_blend_gpu;
    torch::Tensor m_img_gpu;
    TriLinearWS m_trilinear_ws;

    std::vector<float> m_img_chw;
    std::vector<float> m_out_chw;

    // 私有子函数
    bool load_lut_binary(const std::string& path, std::vector<float>& lut);
    void build_adaptive_lut(const std::vector<float>& lut0,
        const std::vector<float>& lut1,
        const std::vector<float>& lut2,
        const float* weights,
        std::vector<float>& final_lut);

    void frame_to_chw_float(const cv::Mat& bgr, std::vector<float>& chw, int& W, int& H);
    cv::Mat chw_float_to_frame(const float* chw, int W, int H);

    torch::Tensor TriLinearApplyGpu(const torch::Tensor& lut_blend,
        const torch::Tensor& img_gpu,
        int dim, int shift, float binsize,
        int W, int H, TriLinearWS& ws);
};

// ================= Impl构造 =================
Lut3DEnhance::Impl::Impl(const std::string& model_dir)
    : m_model_dir(model_dir)
{
    m_final_lut.resize(LUT_SIZE);
}

// ================= CPU插值实现 =================
void Lut3DEnhance::Impl::TriLinearForwardCpu(const float* lut, const float* image, float* output,
    int dim, int shift, float binsize,
    int width, int height, int channels)
{
    const int npixels = height * width;
#pragma omp parallel for simd
    for (int idx = 0; idx < npixels; ++idx)
    {
        float r = image[idx];
        float g = image[idx + npixels];
        float b = image[idx + npixels * 2];

        if (r >= 1.0f) r = 0.999999f;
        if (g >= 1.0f) g = 0.999999f;
        if (b >= 1.0f) b = 0.999999f;

        int r_id = static_cast<int>(r / binsize);
        int g_id = static_cast<int>(g / binsize);
        int b_id = static_cast<int>(b / binsize);

        float r_d = fmodf(r, binsize) / binsize;
        float g_d = fmodf(g, binsize) / binsize;
        float b_d = fmodf(b, binsize) / binsize;

        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = id000 + 1;
        int id010 = id000 + dim;
        int id110 = id000 + dim + 1;
        int id001 = id000 + dim * dim;
        int id101 = id000 + dim * dim + 1;
        int id011 = id000 + dim * dim + dim;
        int id111 = id000 + dim * dim + dim + 1;

        float w000 = (1 - r_d) * (1 - g_d) * (1 - b_d);
        float w100 = r_d * (1 - g_d) * (1 - b_d);
        float w010 = (1 - r_d) * g_d * (1 - b_d);
        float w110 = r_d * g_d * (1 - b_d);
        float w001 = (1 - r_d) * (1 - g_d) * b_d;
        float w101 = r_d * (1 - g_d) * b_d;
        float w011 = (1 - r_d) * g_d * b_d;
        float w111 = r_d * g_d * b_d;

        output[idx] =
            w000 * lut[id000] + w100 * lut[id100] +
            w010 * lut[id010] + w110 * lut[id110] +
            w001 * lut[id001] + w101 * lut[id101] +
            w011 * lut[id011] + w111 * lut[id111];

        output[idx + npixels] =
            w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
            w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
            w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
            w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

        output[idx + npixels * 2] =
            w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] +
            w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] +
            w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] +
            w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    }
}

// ================= 子函数实现 =================
bool Lut3DEnhance::Impl::load_lut_binary(const std::string& path, std::vector<float>& lut)
{
    lut.resize(LUT_SIZE);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open LUT file: " << path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(lut.data()), LUT_SIZE * sizeof(float));
    if (file.gcount() != static_cast<std::streamsize>(LUT_SIZE * sizeof(float))) {
        std::cerr << "Error: LUT file size mismatch: " << path << std::endl;
        return false;
    }
    file.close();
    return true;
}

void Lut3DEnhance::Impl::build_adaptive_lut(const std::vector<float>& lut0,
    const std::vector<float>& lut1,
    const std::vector<float>& lut2,
    const float* weights,
    std::vector<float>& final_lut)
{
    final_lut.resize(LUT_SIZE);
    const float w0 = weights[0];
    const float w1 = weights[1];
    const float w2 = weights[2];
    for (int i = 0; i < LUT_SIZE; ++i) {
        final_lut[i] = w0 * lut0[i] + w1 * lut1[i] + w2 * lut2[i];
    }
}

void Lut3DEnhance::Impl::frame_to_chw_float(const cv::Mat& bgr, std::vector<float>& chw, int& W, int& H)
{
    W = bgr.cols;
    H = bgr.rows;
    const int npixels = W * H;
    size_t total_elem = 3ULL * npixels;
    if (chw.size() < total_elem)
        chw.resize(total_elem);

    for (int y = 0; y < H; ++y) {
        const uint8_t* row = bgr.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float b = row[x * 3 + 0] / 255.0f;
            float g = row[x * 3 + 1] / 255.0f;
            float r = row[x * 3 + 2] / 255.0f;
            chw[idx] = r;
            chw[idx + npixels] = g;
            chw[idx + npixels * 2] = b;
        }
    }
}

cv::Mat Lut3DEnhance::Impl::chw_float_to_frame(const float* chw, int W, int H)
{
    cv::Mat bgr(H, W, CV_8UC3);
    const int npixels = W * H;
    for (int y = 0; y < H; ++y) {
        uint8_t* row = bgr.ptr<uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            int idx = y * W + x;
            float r = chw[idx];
            float g = chw[idx + npixels];
            float b = chw[idx + npixels * 2];

            auto to_u8 = [](float v) -> uint8_t {
                int iv = static_cast<int>(v * 255.0f + 0.5f);
                if (iv < 0) iv = 0;
                if (iv > 255) iv = 255;
                return static_cast<uint8_t>(iv);
            };

            row[x * 3 + 0] = to_u8(b);
            row[x * 3 + 1] = to_u8(g);
            row[x * 3 + 2] = to_u8(r);
        }
    }
    return bgr;
}

torch::Tensor Lut3DEnhance::Impl::TriLinearApplyGpu(const torch::Tensor& lut_blend,
    const torch::Tensor& img_gpu,
    int dim, int shift, float binsize,
    int W, int H, TriLinearWS& ws)
{
    int N = H * W, lut_total = 3 * shift;
    ws.ensure(H, W, img_gpu.device());

    auto c = img_gpu.clamp(0.0f, 0.999f);
    auto r = c[0], g = c[1], b = c[2];
    auto ri = r / binsize, gi = g / binsize, bi = b / binsize;
    auto rid = ri.to(torch::kLong), gid = gi.to(torch::kLong), bid = bi.to(torch::kLong);
    auto rd = ri - rid.to(torch::kFloat32);
    auto gd = gi - gid.to(torch::kFloat32);
    auto bd = bi - bid.to(torch::kFloat32);

    std::vector<torch::Tensor> corners(8);
    corners[0] = (rid + 0 + (gid + 0) * dim + (bid + 0) * dim * dim).flatten();
    corners[1] = (rid + 1 + (gid + 0) * dim + (bid + 0) * dim * dim).flatten();
    corners[2] = (rid + 0 + (gid + 1) * dim + (bid + 0) * dim * dim).flatten();
    corners[3] = (rid + 1 + (gid + 1) * dim + (bid + 0) * dim * dim).flatten();
    corners[4] = (rid + 0 + (gid + 0) * dim + (bid + 1) * dim * dim).flatten();
    corners[5] = (rid + 1 + (gid + 0) * dim + (bid + 1) * dim * dim).flatten();
    corners[6] = (rid + 0 + (gid + 1) * dim + (bid + 1) * dim * dim).flatten();
    corners[7] = (rid + 1 + (gid + 1) * dim + (bid + 1) * dim * dim).flatten();
    for (int i = 0; i < 8; i++) ws.idx_buf[i].copy_(corners[i]);

    auto w000 = (1 - rd) * (1 - gd) * (1 - bd), w100 = rd * (1 - gd) * (1 - bd);
    auto w010 = (1 - rd) * gd * (1 - bd), w110 = rd * gd * (1 - bd);
    auto w001 = (1 - rd) * (1 - gd) * bd, w101 = rd * (1 - gd) * bd;
    auto w011 = (1 - rd) * gd * bd, w111 = rd * gd * bd;
    std::vector<torch::Tensor> weights = {
        w000.flatten(), w100.flatten(), w010.flatten(), w110.flatten(),
        w001.flatten(), w101.flatten(), w011.flatten(), w111.flatten()
    };
    for (int i = 0; i < 8; i++) ws.w_buf[i].copy_(weights[i]);

    auto out_r = lut_blend.take((ws.idx_buf).clamp(0, lut_total - 1))
        .mul(ws.w_buf).sum(0).view({ H, W });
    auto out_g = lut_blend.take((ws.idx_buf + shift).clamp(0, lut_total - 1))
        .mul(ws.w_buf).sum(0).view({ H, W });
    auto out_b = lut_blend.take((ws.idx_buf + 2 * shift).clamp(0, lut_total - 1))
        .mul(ws.w_buf).sum(0).view({ H, W });
    return torch::stack({ out_r, out_g, out_b }, 0);
}

bool Lut3DEnhance::Impl::init()
{
    std::string lut0_path = m_model_dir + "/LUT0_sRGB.bin";
    std::string lut1_path = m_model_dir + "/LUT1_sRGB.bin";
    std::string lut2_path = m_model_dir + "/LUT2_sRGB.bin";
    std::string cls_path = m_model_dir + "/classifier_sRGB.pt";
    auto exists = [](const std::string& p) { return std::filesystem::exists(p); };
    if (!exists(lut0_path) || !exists(lut1_path) || !exists(lut2_path) || !exists(cls_path))
    {
        std::cerr << "Model file missing!" << std::endl;
        return false;
    }

    if (!load_lut_binary(lut0_path, m_lut0)) return false;
    if (!load_lut_binary(lut1_path, m_lut1)) return false;
    if (!load_lut_binary(lut2_path, m_lut2)) return false;

    try {
        m_use_cuda = torch::cuda::is_available();
    }
    catch (...) { m_use_cuda = false; }
    m_device = m_use_cuda ? torch::kCUDA : torch::kCPU;

    try {
        m_classifier = torch::jit::load(cls_path, torch::kCPU);
        m_classifier.eval();
        if (m_use_cuda) {
            try {
                m_classifier.to(torch::kCUDA);
            }
            catch (...) {
                m_use_cuda = false;
                m_device = torch::kCPU;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Load classifier failed: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "Load classifier failed: unknown error" << std::endl;
        return false;
    }

    if (m_use_cuda)
    {
        auto cpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto gpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(m_device);
        m_lut0_gpu = torch::from_blob(m_lut0.data(), { LUT_SIZE }, cpu_opt).clone().to(m_device);
        m_lut1_gpu = torch::from_blob(m_lut1.data(), { LUT_SIZE }, cpu_opt).clone().to(m_device);
        m_lut2_gpu = torch::from_blob(m_lut2.data(), { LUT_SIZE }, cpu_opt).clone().to(m_device);
        m_lut_blend_gpu = torch::empty_like(m_lut0_gpu);
    }

    std::cout << "Processor init success, run on " << (m_use_cuda ? "GPU" : "CPU") << std::endl;
    return true;
}

cv::Mat Lut3DEnhance::Impl::process(const cv::Mat& input_bgr, int out_w, int out_h)
{
    if (input_bgr.empty()) return cv::Mat();
    cv::Mat src = input_bgr.clone();
    int W = src.cols, H = src.rows;
    if (out_w > 0 && out_h > 0)
    {
        cv::resize(src, src, cv::Size(out_w, out_h));
        W = out_w; H = out_h;
    }
    cv::Mat out_frame;

    if (m_use_cuda)
    {
        torch::NoGradGuard no_grad;
        auto bgr_cpu = torch::from_blob(src.data, { H, W, 3 },
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        auto gpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(m_device);
        if (!m_img_gpu.defined() || m_img_gpu.size(1) != H || m_img_gpu.size(2) != W)
        {
            m_img_gpu = torch::empty({ 3, H, W }, gpu_opt);
        }
        auto temp_img = bgr_cpu.to(m_device).permute({ 2, 0, 1 })
            .flip({ 0 })
            .to(torch::kFloat32).div_(255.0f);
        m_img_gpu.copy_(temp_img);

        auto pred = m_classifier.forward({ m_img_gpu.unsqueeze(0) })
            .toTensor().squeeze().to(torch::kCPU);
        float w0 = pred[0].item<float>();
        float w1 = pred[1].item<float>();
        float w2 = pred[2].item<float>();

        m_lut_blend_gpu.copy_(m_lut0_gpu).mul_(w0);
        m_lut_blend_gpu.add_(m_lut1_gpu * w1);
        m_lut_blend_gpu.add_(m_lut2_gpu * w2);

        auto result_gpu = TriLinearApplyGpu(m_lut_blend_gpu, m_img_gpu,
            LUT_DIM, LUT_SHIFT, BINSIZE, W, H, m_trilinear_ws);

        auto out_bgr = result_gpu.flip({ 0 })
            .permute({ 1, 2, 0 })
            .mul_(255.0f).clamp_(0, 255)
            .to(torch::kUInt8)
            .contiguous()
            .to(torch::kCPU);
        cv::Mat temp_out(H, W, CV_8UC3, out_bgr.data_ptr<uint8_t>());
        out_frame = temp_out.clone();
    }
    else
    {
        torch::NoGradGuard no_grad;
        frame_to_chw_float(src, m_img_chw, W, H);
        size_t npixels = (size_t)H * W;
        if (m_out_chw.size() < 3 * npixels)
            m_out_chw.resize(3 * npixels);
        auto cpu_opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto input = torch::from_blob(m_img_chw.data(), { 1, 3, H, W }, cpu_opt).clone();
        auto pred = m_classifier.forward({ input }).toTensor().squeeze();
        float weights[3] = { pred[0].item<float>(), pred[1].item<float>(), pred[2].item<float>() };
        build_adaptive_lut(m_lut0, m_lut1, m_lut2, weights, m_final_lut);

        TriLinearForwardCpu(m_final_lut.data(), m_img_chw.data(), m_out_chw.data(),
            LUT_DIM, LUT_SHIFT, BINSIZE, W, H, 3);

        out_frame = chw_float_to_frame(m_out_chw.data(), W, H);
    }
    return out_frame;
}

// ===================== 外层Lut3DEnhance接口实现 =====================
Lut3DEnhance::Lut3DEnhance(const std::string& model_path)
    : pImpl(new Impl(model_path))
{
    // 构造自动初始化
    if (!pImpl->init())
    {
        throw std::runtime_error("Lut3DEnhance init failed, check model path");
    }
}

Lut3DEnhance::~Lut3DEnhance()
{
    delete pImpl;
}

cv::Mat Lut3DEnhance::process(const cv::Mat& src_bgr, cv::Size out_size)
{
    return pImpl->process(src_bgr, out_size.width, out_size.height);
}