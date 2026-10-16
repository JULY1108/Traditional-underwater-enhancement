#include "smooth_clahe.h"
#include <iostream>

// 辅助宏：计算 LUT 线性索引
// idx = (row * gridX + col) * 256 + bin_value
#define LUT_IDX(gx, gy, val) (((gy) * gridX_ + (gx)) * 256 + (val))

SmoothCLAHE::SmoothCLAHE(double clipLimit, cv::Size tileGridSize, bool enableLUTFilter, bool useBilinear)
    : clipLimit_(clipLimit), tileSize_(tileGridSize), filterEnable_(enableLUTFilter), useBilinear_(useBilinear) {}


void SmoothCLAHE::apply(const cv::Mat &src, cv::Mat &dst) { 
    CV_Assert(src.type() == CV_8UC1);
    dst.create(src.size(), src.type());

    // 1. 计算网格数量
    gridX_ = std::max(1, src.cols / tileSize_.width);
    gridY_ = std::max(1, src.rows / tileSize_.height);

    // 2. 分配连续内存用于 LUT
    const int totalLUTSize = gridY_ * gridX_ * 256;
    std::vector<float> lutBuffer(totalLUTSize, 0.0f);

    // 3. 计算直方图并生成 LUT
    calculateMapsOptimized(src, lutBuffer);

    // 4. 滤波
    if (filterEnable_) {
        filterLUTsOptimized(lutBuffer);
    }

    // 5. 预计算 X 轴插值参数
    //std::vector<int> ind1(src.cols);
    //std::vector<int> ind2(src.cols);
    //std::vector<float> xa(src.cols);
    //std::vector<float> xa1(src.cols);
    ind1_.resize(src.cols);
    ind2_.resize(src.cols);
    xa_.resize(src.cols);
    xa1_.resize(src.cols);

    float inv_tw = 1.0f / tileSize_.width;

    for (int x = 0; x < src.cols; x++) {
        float txf = (float)x * inv_tw - 0.5f;
        int tx1 = cvFloor(txf);
        int tx2 = tx1 + 1;

        xa_[x] = txf - tx1;
        xa1_[x] = 1.0f - xa_[x];

        // 边界 Clamp
        tx1 = std::max(0, std::min(tx1, gridX_ - 1));
        tx2 = std::max(0, std::min(tx2, gridX_ - 1));

        ind1_[x] = tx1 * 256;
        ind2_[x] = tx2 * 256;
    }

    // 6. 主处理循环
    float inv_th = 1.0f / tileSize_.height;
    const float *lutPtr = lutBuffer.data();

    for (int i = 0; i < src.rows; i++) {
        float tyf = (float)i * inv_th - 0.5f;
        int ty1 = cvFloor(tyf);
        int ty2 = ty1 + 1;

        float ya = tyf - ty1;
        float ya1 = 1.0f - ya;

        ty1 = std::max(0, std::min(ty1, gridY_ - 1));
        ty2 = std::max(0, std::min(ty2, gridY_ - 1));

        // 计算行偏移
        int rowOffset1 = ty1 * gridX_ * 256;
        int rowOffset2 = ty2 * gridX_ * 256;

        const uchar *srcRow = src.ptr<uchar>(i);
        uchar *dstRow = dst.ptr<uchar>(i);

        for (int j = 0; j < src.cols; j++) {
            uchar val = srcRow[j];

            if (useBilinear_) {
                // --- 双线性插值逻辑 ---
                const float *p11 = lutPtr + rowOffset1 + ind1_[j] + val;
                const float *p12 = lutPtr + rowOffset1 + ind2_[j] + val;
                const float *p21 = lutPtr + rowOffset2 + ind1_[j] + val;
                const float *p22 = lutPtr + rowOffset2 + ind2_[j] + val;

                float v1 = (*p11) * xa1_[j] + (*p12) * xa_[j];
                float v2 = (*p21) * xa1_[j] + (*p22) * xa_[j];
                float result = v1 * ya1 + v2 * ya;

                dstRow[j] = (uchar)(result + 0.5f);
            } else {
                // --- 新增的最近邻插值逻辑 (关闭双线性) ---
                // 直接取左上角块 (ty1, tx1) 的 LUT 值，不做任何混合
                int tx1_idx = ind1_[j] / 256; // 还原回 tx1 的索引
                //// 这里为了严谨，重新做一次 clamp 确保最近邻取的也是合法边界内的块
                //tx1_idx = std::max(0, std::min(tx1_idx, gridX_ - 1));
                //ty1 = std::max(0, std::min(ty1, gridY_ - 1));

                int nearestOffset = ty1 * gridX_ * 256 + tx1_idx * 256 + val;
                dstRow[j] = (uchar)(lutPtr[nearestOffset] + 0.5f);
            }
        }
    }
}


// 计算 Maps，直接写入 float 数组
void SmoothCLAHE::calculateMapsOptimized(const cv::Mat &src, std::vector<float> &lutBuffer) {
    int clipLimitVal = (int)(clipLimit_ * tileSize_.area() / 256.0);
    if (clipLimitVal < 1)
        clipLimitVal = 1;

    for (int gy = 0; gy < gridY_; ++gy) {
        for (int gx = 0; gx < gridX_; ++gx) {
            int x_start = gx * tileSize_.width;
            int y_start = gy * tileSize_.height;
            int width = std::min(tileSize_.width, src.cols - x_start);
            int height = std::min(tileSize_.height, src.rows - y_start);

            // 边界检查
            if (width <= 0 || height <= 0)
                continue;

            cv::Mat tile = src(cv::Rect(x_start, y_start, width, height));

            // 1. 计算直方图 (使用指针遍历加速)
            int hist[256] = {0};
            for (int y = 0; y < height; ++y) {
                const uchar *ptr = tile.ptr<uchar>(y);
                for (int x = 0; x < width; ++x) {
                    hist[ptr[x]]++;
                }
            }

            // 2. 裁剪与重新分配
            int excess = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clipLimitVal) {
                    excess += hist[i] - clipLimitVal;
                    hist[i] = clipLimitVal;
                }
            }
            int redist = excess / 256;
            int remainder = excess % 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += redist;
                if (i < remainder)
                    hist[i]++;
            }

            // 3. 直接写入 LUT Buffer
            float scale = 255.0f / (width * height);
            int sum = 0;
            // 计算当前 Tile 在 Buffer 中的基地址
            float *tileLUT = lutBuffer.data() + (gy * gridX_ + gx) * 256;

            for (int i = 0; i < 256; ++i) {
                sum += hist[i];
                tileLUT[i] = sum * scale;
            }
        }
    }
}

// 优化版：滤波，直接操作 float 数组
void SmoothCLAHE::filterLUTsOptimized(std::vector<float> &lutBuffer) {
    // 使用临时 Buffer 存储结果，避免读写冲突
    std::vector<float> tempBuffer(lutBuffer.size(), 0.0f);

    // 高斯核
    const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    const float kernelScale = 1.0f / 16.0f;

    for (int gy = 0; gy < gridY_; ++gy) {
        for (int gx = 0; gx < gridX_; ++gx) {
            float *destLUT = tempBuffer.data() + (gy * gridX_ + gx) * 256;

            // 对当前 Tile 的 256 个 bin 进行循环
            for (int bin = 0; bin < 256; ++bin) {
                float sumVal = 0.0f;

                // 3x3 邻域卷积
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ny = gy + ky;
                        int nx = gx + kx;

                        // 边界镜像/Clamp
                        ny = std::max(0, std::min(ny, gridY_ - 1));
                        nx = std::max(0, std::min(nx, gridX_ - 1));

                        const float *srcLUT = lutBuffer.data() + (ny * gridX_ + nx) * 256;
                        sumVal += srcLUT[bin] * kernel[ky + 1][kx + 1];
                    }
                }
                destLUT[bin] = sumVal * kernelScale;
            }
        }
    }

    // 拷贝最终的结果
    lutBuffer.swap(tempBuffer);
}

void SmoothCLAHE::calculateMaps(const cv::Mat &src, std::vector<std::vector<cv::Mat>> &maps) {
    // 每个 bin 的像素数上限
    int clipLimit = (int)(clipLimit_ * tileSize_.area() / 256.0);
    if (clipLimit < 1)
        clipLimit = 1;

    for (int gy = 0; gy < gridY_; ++gy) {
        for (int gx = 0; gx < gridX_; ++gx) {
            // 计算当前 Tile 的 ROI
            int x_start = gx * tileSize_.width;
            int y_start = gy * tileSize_.height;
            int width = std::min(tileSize_.width, src.cols - x_start);
            int height = std::min(tileSize_.height, src.rows - y_start);

            cv::Mat tile = src(cv::Rect(x_start, y_start, width, height));

            // 1. 计算直方图
            int hist[256] = {0};
            for (int y = 0; y < height; ++y) {
                const uchar *ptr = tile.ptr<uchar>(y);
                for (int x = 0; x < width; ++x) {
                    hist[ptr[x]]++;
                }
            }

            // 2. 裁剪直方图并重新分配 (Clip and Redistribute)
            int excess = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clipLimit) {
                    excess += hist[i] - clipLimit;
                    hist[i] = clipLimit;
                }
            }
            // 均匀分配多余像素
            int redist = excess / 256;
            int remainder = excess % 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += redist;
                if (i < remainder)
                    hist[i]++;
            }

            // 3. 生成累积分布函数 (CDF) -> 即 LUT
            maps[gy][gx] = cv::Mat::zeros(256, 1, CV_32F);
            float scale = 255.0f / (width * height); // 归一化到 0-255
            int sum = 0;
            for (int i = 0; i < 256; ++i) {
                sum += hist[i];
                maps[gy][gx].at<float>(i) = sum * scale;
            }
        }
    }
}

void SmoothCLAHE::filterLUTs(std::vector<std::vector<cv::Mat>> &maps) {
    int rows = maps.size();
    int cols = maps[0].size();

    std::vector<std::vector<cv::Mat>> temp_maps = maps;

    // 3x3 加权平均核
    // 可能会被背景的直方图污染
    // 1 1 1
    // 1 8 1  (中心权重高，保留细节)
    // 1 1 1
    // const int kernel[3][3] = {{1, 1, 1}, {1, 8, 1}, {1, 1, 1}};

    // 高斯滤波核，测试两者差别不大
    const int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    const int kernel_sum = 16;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 临时变量，用于累加加权后的 LUT
            cv::Mat temp_mat;
            cv::Mat sumLUT = cv::Mat::zeros(256, 1, CV_32F);

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;

                    // 边界镜像处理
                    if (ni < 0)
                        ni = 0;
                    if (ni >= rows)
                        ni = rows - 1;
                    if (nj < 0)
                        nj = 0;
                    if (nj >= cols)
                        nj = cols - 1;

                    // 确保是 float 类型进行计算
                    if (temp_maps[ni][nj].type() != CV_32F) {
                        temp_maps[ni][nj].convertTo(temp_mat, CV_32F);
                    } else {
                        temp_mat = temp_maps[ni][nj];
                    }

                    // 加权累加
                    sumLUT += temp_mat * kernel[ki + 1][kj + 1];
                }
            }

            // 归一化并保存回 maps
            maps[i][j] = sumLUT / kernel_sum;
        }
    }
}

