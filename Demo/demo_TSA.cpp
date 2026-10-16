#include <opencv2/opencv.hpp>
#include "dehaze_common.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: demo_images <input_directory> <new_output_directory>\n";
        return 1;
    }
    try {
        const fs::path input = fs::canonical(argv[1]);
        const fs::path output = fs::absolute(argv[2]);
        if (!fs::is_directory(input))
            throw std::runtime_error("Input must be a directory.");
        // Require a new directory to avoid overwriting existing images.
        if (!fs::create_directory(output))
            throw std::runtime_error("Output directory already exists.");
        std::vector<fs::path> images;
        for (const auto& entry : fs::directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                ext == ".bmp" || ext == ".tif" || ext == ".tiff")
                images.push_back(entry.path());
        }
        if (images.empty()) throw std::runtime_error("No supported images found.");
        std::sort(images.begin(), images.end());
        Rank1PlusUnderwater enhancer;
        for (const auto& image : images) {
            const cv::Mat src = cv::imread(image.string());
            if (src.empty()) throw std::runtime_error("Cannot read: " + image.string());
            const cv::Mat result = enhancer.process(src, 1.0);
            if (!cv::imwrite((output / image.filename()).string(), result))
                throw std::runtime_error("Cannot write: " + image.filename().string());
            std::cout << image.filename().string() << '\n';
        }
        std::cout << "Processed " << images.size() << " images.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

