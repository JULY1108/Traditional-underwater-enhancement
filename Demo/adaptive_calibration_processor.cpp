#include <opencv2/opencv.hpp>

#include "adaptive_calibration_processor.h"

#include <torch/cuda.h>
#include <torch/script.h>

#include <iostream>
#include <stdexcept>

class AdaptiveCalibrationEnhance::Impl
{
public:
    explicit Impl(const std::string& model_path)
        : model_path_(model_path)
    {
        device_ = torch::cuda::is_available()
            ? torch::Device(torch::kCUDA)
            : torch::Device(torch::kCPU);

        try
        {
            model_ = torch::jit::load(model_path_, device_);
            model_.eval();
            std::cout << "Adaptive calibration model loaded on "
                      << (device_.is_cuda() ? "CUDA" : "CPU")
                      << ": " << model_path_ << std::endl;
        }
        catch (const c10::Error& error)
        {
            throw std::runtime_error(
                "Failed to load adaptive calibration TorchScript model '" +
                model_path_ + "': " + error.what_without_backtrace());
        }
    }

    cv::Mat process(const cv::Mat& src_bgr)
    {
        CV_Assert(src_bgr.type() == CV_8UC3);

        // Training uses RGB float32 tensors in [0, 1]; OpenCV stores BGR.
        cv::Mat rgb;
        cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);
        auto input = torch::from_blob(
                         rgb.data,
                         {1, rgb.rows, rgb.cols, 3},
                         torch::TensorOptions().dtype(torch::kUInt8))
                         .permute({0, 3, 1, 2})
                         .to(device_, torch::kFloat32)
                         .div(255.0);

        torch::NoGradGuard no_grad;
        const auto output = model_.forward({input}).toTensor();
        auto output_u8 = output
                             .to(torch::kCPU)
                             .clamp(0.0, 1.0)
                             .squeeze(0)
                             .permute({1, 2, 0})
                             .mul(255.0)
                             .to(torch::kU8)
                             .contiguous();

        cv::Mat rgb_output(
            src_bgr.rows,
            src_bgr.cols,
            CV_8UC3,
            output_u8.data_ptr<unsigned char>());
        cv::Mat bgr_output;
        cv::cvtColor(rgb_output, bgr_output, cv::COLOR_RGB2BGR);
        return bgr_output.clone();
    }

private:
    std::string model_path_;
    torch::jit::Module model_;
    torch::Device device_ = torch::Device(torch::kCPU);
};

AdaptiveCalibrationEnhance::AdaptiveCalibrationEnhance(
    const std::string& model_path)
    : pImpl(new Impl(model_path))
{}

AdaptiveCalibrationEnhance::~AdaptiveCalibrationEnhance()
{
    delete pImpl;
}

cv::Mat AdaptiveCalibrationEnhance::process(const cv::Mat& src_bgr)
{
    return pImpl->process(src_bgr);
}
