# Traditional underwater enhancement

C++17 / OpenCV 水下图像批量增强程序。

当前实际启用的算法是 **逐通道 SmoothCLAHE → 通道归一化 → 带高光/暗部保护的联合百分位拉伸**。保留原有参数与实现；当前路径未启用物理去雾或神经网络。

## 构建

需要 CMake 3.14+、C++17 编译器、OpenCV 4 和 OpenMP。Windows 建议使用 Visual Studio 2019 或更新版本的 x64 开发者命令行。

```sh
cmake -S . -B build -DOpenCV_DIR="你的OpenCV配置目录"
cmake --build build --config Release
```

Windows 运行前将 OpenCV DLL 所在的 bin 目录加入 PATH；Linux 同样需要可用的 OpenCV 动态库。

## 使用

```sh
# Windows 多配置构建
build/Demo/Release/demo_images.exe input_images output_images
# Linux 单配置构建
./build/Demo/demo_images input_images output_images
```

输入为图片文件夹，支持 jpg/jpeg/png/bmp/tif/tiff（不递归），输出保留文件名。输出目录的父目录必须存在，输出目录本身必须不存在，避免覆盖已有图片。读取或写入失败返回非零状态；之前成功输出的图片保留。建议使用 ASCII 路径，Windows 上 OpenCV 对中文路径的支持取决于版本。

## 文件说明

- `Demo/demo_TSA.cpp`：批量处理入口，路径通过命令行传入。
- `Demo/dehaze_old.cpp`：当前实际使用的增强实现，保留原文件名。
- `Demo/dehaze_common.h`：增强接口。
- `Demo/smooth_clahe.*`：平滑 CLAHE 实现。
- `Demo/lut_batch_processor.*`、`Demo/adaptive_calibration_processor.*`：保留的可选 LibTorch 模块，默认构建不启用，需要另行准备匹配的模型和 LibTorch。
- `Demo/exported_models/model_info.txt`：原 LUT 格式说明。模型权重不随本次源码上传。

原 Visual Studio 工程和本机脚本含机器专属依赖路径，以此处 CMake 构建为准。全注释的历史文件、临时副本、编译产物、原始视频和输出图片未纳入发布源码。
