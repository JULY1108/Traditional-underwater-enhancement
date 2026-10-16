# Traditional Underwater Enhancement

水下图像与视频增强项目，使用 C++、OpenCV 和 OpenMP 实现传统增强处理，并提供基于 LibTorch 的 3D LUT 调色与自适应校准模块。

## 项目内容

- **传统增强**：逐通道 SmoothCLAHE、通道归一化，以及带高光和暗部保护的联合百分位拉伸。
- **3D LUT 调色**：通过分类器预测三个 LUT 的混合权重，使用三线性插值完成颜色映射，支持 CPU / CUDA 路径。
- **自适应校准**：加载 TorchScript 模型进行 RGB 图像校准。
- **测试示例**：`demo_video.cpp` 提供视频逐帧处理示例；`demo_TSA.cpp` 保留原始批量图片示例备份。

当前 `Rank1PlusUnderwater::process()` 的实际处理链为：

```text
输入 BGR 图像 → 按比例缩放 → 逐通道 SmoothCLAHE
             → 通道归一化 → 联合百分位拉伸 → 输出 BGR 图像
```

当前实现未启用物理去雾步骤。模块名称沿用原代码命名。

## 目录结构

```text
Traditional-underwater-enhancement/
├── README.md
├── CMakeLists.txt
└── Demo/
    ├── CMakeLists.txt
    ├── demo_video.cpp                    # 视频测试示例
    ├── demo_TSA.cpp                      # 批量图片示例备份（当前全部注释）
    ├── dehaze_common.h                   # 传统增强接口
    ├── dehaze_old.cpp                    # 当前传统增强实现
    ├── smooth_clahe.h / .cpp             # 平滑 CLAHE
    ├── lut_batch_processor.h / .cpp      # 3D LUT 调色模块
    ├── adaptive_calibration_processor.h / .cpp
    └── exported_models/                  # 已上传的模型与 LUT 数据
```

## 环境依赖

| 依赖 | 用途 |
| --- | --- |
| C++17 编译器 | `std::filesystem` 等语言与标准库功能 |
| OpenCV 4 | 图像处理、视频读写和预览窗口 |
| OpenMP | 并行处理 |
| LibTorch | LUT 分类器与自适应校准模型推理 |
| CUDA（可选） | 使用兼容的 GPU 版本 LibTorch 时加速推理 |

传统增强核心使用 OpenCV、OpenMP 和标准线程库。当前视频示例会构造 `Lut3DEnhance`，即使 LUT 调色调用被注释，仍需要配置 LibTorch 并提供对应模型文件。

## 视频测试示例

示例文件：[Demo/demo_video.cpp](Demo/demo_video.cpp)。

运行前，在示例源码中按实际环境设置：

| 变量 | 含义 |
| --- | --- |
| `input_video_path` | 输入视频的完整路径 |
| `out_folder` | 输出视频目录 |
| `enhance_video_name` | 输出视频文件名 |
| `resize_scale` | 图像缩放比例，当前为 `1` |

模型目录使用相对路径 `./exported_models`，因此运行时应将工作目录设为 `Demo`，或将 `exported_models` 放到程序工作目录下。路径配置写在源码中，当前示例不接收命令行路径参数。

示例逐帧读取视频，调用传统增强接口，显示增强预览，并使用 `mp4v` 编码保存结果；按 `Esc` 可提前结束。运行时会输出逐帧耗时、处理进度和平均 FPS。视频读写需要 OpenCV 具备可用的视频编解码后端，预览需要图形界面环境。

当前视频示例的 LUT 调色调用被注释，输出为传统增强结果；自适应校准模块未接入该示例。

### 构建说明

仓库当前的 CMake 配置仅包含传统增强库及指向 `demo_TSA.cpp` 的 `demo_images` 目标，**尚未配置视频示例的构建，也未链接 LibTorch**。由于图片示例备份全部被注释，直接构建 `demo_images` 会缺少 `main()`，不能作为开箱即用的运行方式。

在自己的 C++ 工程中运行视频示例时，应选择 `demo_video.cpp` 作为唯一入口，并加入：

```text
Demo/demo_video.cpp
Demo/dehaze_old.cpp
Demo/smooth_clahe.cpp
Demo/lut_batch_processor.cpp
```

配置 OpenCV、LibTorch、OpenMP 和线程库的头文件、链接库与运行时动态库路径。OpenCV 需要支持图像处理、视频读写和窗口显示；使用 CUDA 版 LibTorch 时还需匹配的 CUDA 运行环境。源码中部分文件使用 GBK 编码，编译器字符集设置需与实际文件编码匹配。

## 批量图片示例

[Demo/demo_TSA.cpp](Demo/demo_TSA.cpp) 按备份原样保留，当前整份文件处于注释状态。示例展示图片目录遍历、传统增强、3D LUT 调色、结果保存及耗时统计，支持 jpg、jpeg、png、bmp、tif 和 tiff。

如需运行该示例，需要自行启用代码、配置其中的输入输出路径和依赖，并确保工程仅有一个有效的 `main()`。

## 模型文件

`Demo/exported_models` 已包含以下文件：

| 文件 | 用途 |
| --- | --- |
| `classifier_sRGB.pt` | 预测三个 LUT 混合权重的分类器 |
| `LUT0_sRGB.bin` | 第一个基础 LUT |
| `LUT1_sRGB.bin` | 第二个基础 LUT |
| `LUT2_sRGB.bin` | 第三个基础 LUT |
| `LUT_identity_sRGB.bin` | 恒等 LUT 数据 |
| `adaptive_calibration_exp001.ts.pt` | 自适应校准 TorchScript 模型 |
| `model_info.txt` | LUT 数据布局与分类器输入输出说明 |

LUT 使用 sRGB 色彩空间，尺寸为 `33 × 33 × 33`，包含三个通道，数据类型为 `float32`。分类器输入为 `[B, 3, H, W]`、范围 `[0, 1]` 的浮点张量，输出为三个混合权重：

```text
final_LUT = w0 × LUT0 + w1 × LUT1 + w2 × LUT2
```

## 当前验证范围

传统增强核心此前通过了本机编译与合成图像检查。原始视频示例、LUT 模型及自适应校准模型尚未完成端到端运行验证；具体效果和速度取决于输入数据、硬件及依赖配置。
