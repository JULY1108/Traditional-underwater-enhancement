//#include <iostream>
//#include <vector>
//#include <iomanip>
//#include <filesystem>
//#include <chrono>
//#include <omp.h>
//
//// 必须加 OpenCV 总头文件
//#include <opencv2/opencv.hpp>
//
//// 引入拆分后的头文件
//#include "dehaze_common.h"
//#include "lut_batch_processor.h"
//
//
//// 关键：OpenCV 所有类型都在 cv 命名空间
//using namespace std;
//using namespace cv;
//namespace fs = std::filesystem;
//
//
//int main()
//{
//    omp_set_num_threads(omp_get_max_threads());
//
//    Rank1PlusUnderwater dehazeObj;
//    Lut3DEnhance Lut3Dobj("./exported_models");
//
//    string in_dir = "H:\\underwater_test_datasets_raw\\强哥_1";
//    string out_dir = "H:\\underwater_test_datasets_raw\\强哥_2_result";
//
//    if (!fs::exists(out_dir))
//        fs::create_directories(out_dir);
//
//    vector<string> img_paths;
//    vector<string> exts = { "*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff" };
//    for (auto& ext : exts)
//    {
//        vector<string> tmp;
//        glob(in_dir + "/" + ext, tmp, false);
//        img_paths.insert(img_paths.end(), tmp.begin(), tmp.end());
//    }
//
//    if (img_paths.empty())
//    {
//        cout << "【错误】test_input文件夹未找到任何图片！" << endl;
//        return -1;
//    }
//
//    cout << "待处理图片总数：" << img_paths.size() << endl;
//    cout << "OpenMP可用线程：" << omp_get_max_threads() << endl;
//    cout << "输入目录：" << in_dir << "\n输出目录：" << out_dir << "\n" << endl;
//
//    double resize_scale = 0.5;     //缩放因子
//    int total_frames = 0;
//    double total_time_ms = 0.0;
//    auto total_start = chrono::high_resolution_clock::now();
//
//    for (auto& img_path : img_paths)
//    {
//        auto t1 = chrono::high_resolution_clock::now();
//        Mat src = imread(img_path);
//        if (src.empty())
//        {
//            cout << "跳过损坏文件：" << img_path << endl;
//            continue;
//        }
//
//
//
//        Mat enhanced = dehazeObj.process(src, resize_scale);     //透水增强主函数
//
//        enhanced = Lut3Dobj.process(enhanced, cv::Size(0, 0)); //3D LUT调色    //主函数
//
//        fs::path p(img_path);
//        string save_name = p.filename().string();
//        string save_path = out_dir + "/" + save_name;
//
//        imwrite(save_path, enhanced);
//
//        auto t2 = chrono::high_resolution_clock::now();
//        double cost_ms = chrono::duration<double, milli>(t2 - t1).count();
//        total_time_ms += cost_ms;
//        total_frames++;
//
//        cout << "[" << total_frames << "/" << img_paths.size() << "] "
//            << save_name << " | 耗时:" << fixed << setprecision(2) << cost_ms << " ms" << endl;
//    }
//
//    auto total_end = chrono::high_resolution_clock::now();
//    double total_cost_s = chrono::duration<double>(total_end - total_start).count();
//    double avg_ms = total_time_ms / total_frames;
//
//    cout << "\n==================== 批量处理完成 ====================" << endl;
//    cout << "成功处理：" << total_frames << " 张" << endl;
//    cout << "总耗时：" << fixed << setprecision(2) << total_cost_s << " s" << endl;
//    cout << "单张平均耗时：" << fixed << setprecision(2) << avg_ms << " ms" << endl;
//    cout << "结果全部保存至：" << out_dir << endl;
//    cout << "======================================================" << endl;
//
//    return 0;
//}