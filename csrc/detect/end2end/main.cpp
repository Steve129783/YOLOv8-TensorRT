//
// Created by ubuntu on 1/20/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = ghc::filesystem;

struct ClassTable {
    std::vector<std::string> names;
    std::vector<std::vector<unsigned int>> colors;
};

ClassTable load_class_info(const std::string& json_path)
{
    std::ifstream ifs(json_path);

    nlohmann::json j;
    ifs >> j;

    ClassTable table;

    for (auto& item : j)
    {
        table.names.push_back(
            item["name"].get<std::string>()
        );

        table.colors.push_back(
            item["color"].get<std::vector<unsigned int>>()
        );
    }

    return table;
}


int main(int argc, char** argv)
{   
    // 使用
    auto class_table = load_class_info("classes.json");

    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const fs::path    path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat             res, image;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) { //读取帧
            objs.clear(); // 清空上一帧的检测结果
            yolov8->copy_from_Mat(image, size); // 输入图像预处理
            auto start = std::chrono::system_clock::now(); // 前向推理计时开始
            yolov8->infer(); // 网络前向推理
            auto end = std::chrono::system_clock::now(); // 前向推理计时结束
            yolov8->postprocess(objs); // 后处理
            yolov8->draw_objects(image, res, objs, class_table.names, class_table.colors); // 将结果绘制到输出图像
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc); // 打印耗时
            cv::imshow("result", res); // 显示结果
            if (cv::waitKey(10) == 'q') { // 按键退出循环
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, class_table.names, class_table.colors);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
