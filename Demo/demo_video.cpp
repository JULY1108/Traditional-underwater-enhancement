#include <iostream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "dehaze_common.h"
#include "lut_batch_processor.h"
using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// 注意：本文件原为 GBK 编码，之前被有损转成 UTF-8 导致中文内容全部损坏、无法恢复。
// 现已整体重写：功能代码与原来完全一致，注释和输出改为重新编写的中文。
int main()
{
    omp_set_num_threads(omp_get_max_threads());
    Rank1PlusUnderwater dehazeObj;
    Lut3DEnhance Lut3Dobj("./exported_models");

    // TODO: 原视频路径中的中文字符已随编码损坏丢失，请改回你的真实路径
    string input_video_path = "H:\\水下测试数据视频\\10_6.mp4";
    string out_folder = "H:\\水下测试数据视频_1";
    string enhance_video_name = "enhance_10_6.mp4";
    const double resize_scale = 1;

    if (!fs::exists(out_folder))
        fs::create_directories(out_folder);

    string enhance_video_path = out_folder + "/" + enhance_video_name;

    if (!fs::exists(input_video_path) || !fs::is_regular_file(input_video_path))
    {
        cout << "输入视频文件不存在: " << input_video_path << endl;
        return -1;
    }

    fs::path video_fs(input_video_path);
    string video_name = video_fs.filename().string();
    cout << "输入视频: " << video_name << endl;
    cout << "OpenMP线程数: " << omp_get_max_threads() << endl;
    cout << "输入路径: " << input_video_path << endl;
    cout << "输出路径: " << enhance_video_path << "\n" << endl;

    int total_frames = 0;
    auto total_start = chrono::high_resolution_clock::now();
    VideoCapture cap(input_video_path);
    if (!cap.isOpened())
    {
        cout << "OpenCV 无法打开视频文件 (FFmpeg?)" << endl;
        return -1;
    }

    int raw_w = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int raw_h = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    long long total_video_frames = static_cast<long long>(cap.get(CAP_PROP_FRAME_COUNT));
    int dst_w = cvRound(raw_w * resize_scale);
    int dst_h = cvRound(raw_h * resize_scale);

    // 统一使用 mp4v 编码，防止缺少 openh264 时报错
    const int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    VideoWriter writer_enhance;
    writer_enhance.open(enhance_video_path, fourcc, fps, Size(dst_w, dst_h));

    if (!writer_enhance.isOpened())
    {
        cout << "创建视频写入器失败" << endl;
        cap.release();
        return -1;
    }

    cout << "原始尺寸: " << raw_w << "x" << raw_h << "  fps: " << fps << "  总帧数: " << total_video_frames << endl;
    cout << "输出尺寸: " << dst_w << "x" << dst_h << "\n开始处理...\n" << endl;

    long long frame_idx = 0;
    Mat frame;
    while (cap.read(frame))
    {
        frame_idx++;
        auto frame_start = chrono::high_resolution_clock::now();

        // 1 原图缩放（视频处理里不需要缩放可注释此行）
        Mat src_resize;
        resize(frame, src_resize, Size(dst_w, dst_h));

        // 2 去雾
        auto proc_start = chrono::high_resolution_clock::now();
        Mat dehazed = dehazeObj.process(frame, resize_scale);

        // 3 颜色增强
        //Mat final_frame = Lut3Dobj.process(dehazed, Size(0, 0));
        Mat final_frame = dehazed;

        auto proc_end = chrono::high_resolution_clock::now();

        // 4 预览窗口（直接预览增强图像）
        imshow("增强预览", final_frame);
        int key = waitKey(1);
        if (key == 27)
        {
            cout << "\n手动按ESC终止" << endl;
            cap.release();
            writer_enhance.release();
            destroyAllWindows();
            return 0;
        }

        // 5 写入增强视频
        auto write_start = chrono::high_resolution_clock::now();
        writer_enhance.write(final_frame);
        auto write_end = chrono::high_resolution_clock::now();
        total_frames++;

        double proc_ms = chrono::duration<double, milli>(proc_end - proc_start).count();
        double write_ms = chrono::duration<double, milli>(write_end - write_start).count();
        double total_frame_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - frame_start).count();
        double instant_fps = 1000.0 / total_frame_ms;

        // 打印每帧处理时间（每帧都输出）
        cout << "frame:" << frame_idx << " process:" << fixed << setprecision(1) << proc_ms
            << "ms total:" << total_frame_ms << "ms FPS:" << instant_fps << endl;

        // 每50帧打印当前进度
        if (frame_idx % 50 == 0)
        {
            auto now = chrono::high_resolution_clock::now();
            double total_elapse = chrono::duration<double>(now - total_start).count();
            double average_fps = total_frames / total_elapse;
            cout << "进度:" << frame_idx << "/" << total_video_frames
                << "  处理耗时:" << fixed << setprecision(1) << proc_ms
                << "ms 文件写入耗时:" << write_ms << "ms | 瞬时FPS:" << instant_fps
                << " 平均FPS:" << average_fps << "\r";
            cout.flush();
        }
    }

    cap.release();
    writer_enhance.release();
    destroyAllWindows();

    auto total_end = chrono::high_resolution_clock::now();
    double total_cost_s = chrono::duration<double>(total_end - total_start).count();
    double avg_ms = total_frames > 0 ? (chrono::duration<double, milli>(total_end - total_start).count() / total_frames) : 0;
    double final_avg_fps = total_frames / total_cost_s;

    cout << endl;
    cout << "\n==================== 视频处理结束 ====================" << endl;
    cout << "处理帧数：" << total_frames << endl;
    cout << "总耗时：" << fixed << setprecision(2) << total_cost_s << " s" << endl;
    cout << "单帧平均总耗时：" << fixed << setprecision(2) << avg_ms << " ms" << endl;
    cout << "全局平均处理FPS：" << fixed << setprecision(2) << final_avg_fps << endl;
    cout << "保存增强视频路径：" << enhance_video_path << endl;
    cout << "======================================================" << endl;
    return 0;
}