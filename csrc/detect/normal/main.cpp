//
// Created by ubuntu on 1/20/23.
//

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "yolov8.hpp"

#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

#include <nlohmann/json.hpp>

namespace fs = ghc::filesystem;

//--------------------------------------------------
// runtime config
//--------------------------------------------------

struct TRTDetectConfig
{
    std::string engine_path;

    std::string class_json_path;

    std::string input_path;

    int device_id = 0;

    float score_thres = 0.25f;

    float iou_thres = 0.65f;
};

//--------------------------------------------------
// class metadata
//--------------------------------------------------

struct ClassTable
{
    std::vector<std::string> names;

    std::vector<std::vector<unsigned int>> colors;
};

//--------------------------------------------------
// infer config
//--------------------------------------------------

struct InferConfig
{
    float score_thres;

    float iou_thres;
};

//--------------------------------------------------
// load class metadata
//--------------------------------------------------

ClassTable load_class_info(
    const std::string& json_path
)
{
    std::ifstream ifs(json_path);

    if (!ifs.is_open()) {

        fprintf(
            stderr,
            "can not open class json: %s\n",
            json_path.c_str()
        );

        std::abort();
    }

    nlohmann::json j;

    ifs >> j;

    if (!j.is_array()) {

        fprintf(
            stderr,
            "class json must be an array\n"
        );

        std::abort();
    }

    ClassTable table;

    for (const auto& item : j)
    {
        if (!item.contains("name") ||
            !item.contains("color")) {

            fprintf(
                stderr,
                "each class item must contain name and color\n"
            );

            std::abort();
        }

        std::string name =
            item.at("name").get<std::string>();

        std::vector<unsigned int> color =
            item.at("color").get<
            std::vector<unsigned int>
            >();

        if (color.size() != 3)
        {
            fprintf(
                stderr,
                "color must have 3 values\n"
            );

            std::abort();
        }

        table.names.push_back(name);

        table.colors.push_back(color);
    }

    return table;
}

//--------------------------------------------------
// main
//--------------------------------------------------

int main(
    int argc,
    char** argv
)
{
    //--------------------------------------------------
    // config
    //--------------------------------------------------

    TRTDetectConfig trt_config;

    trt_config.engine_path =
        "D:/steve/resources/dep_res/20260530/best.engine";

    trt_config.class_json_path =
        "D:/steve/resources/dep_res/20260530/classes.json";

    trt_config.input_path =
        "D:/steve/resources/dep_res/20260530/test.jpg";

    trt_config.device_id = 0;

    trt_config.score_thres = 0.1f;

    trt_config.iou_thres = 0.1f;

    //--------------------------------------------------
    // semantic alias
    //--------------------------------------------------

    const std::string engine_file_path{
        trt_config.engine_path
    };

    const fs::path path{
        trt_config.input_path
    };

    //--------------------------------------------------
    // cuda device
    //--------------------------------------------------

    cudaError_t cuda_status =
        cudaSetDevice(
            trt_config.device_id
        );

    if (cuda_status != cudaSuccess)
    {
        fprintf(
            stderr,
            "cudaSetDevice failed: %s\n",
            cudaGetErrorString(cuda_status)
        );

        return -1;
    }

    //--------------------------------------------------
    // load class metadata
    //--------------------------------------------------

    ClassTable class_table =
        load_class_info(
            trt_config.class_json_path
        );

    //--------------------------------------------------
    // input path parse
    //--------------------------------------------------

    std::vector<std::string> imagePathList;

    bool isVideo{ false };

    //--------------------------------------------------
    // create runtime
    //--------------------------------------------------

    auto yolov8 =
        new YOLOv8(engine_file_path);

    //--------------------------------------------------
    // runtime semantic metadata
    //--------------------------------------------------

    yolov8->num_labels_ =
        static_cast<int>(
            class_table.names.size()
            );

    std::cout
        << "num labels : "
        << yolov8->num_labels_
        << std::endl;

    yolov8->make_pipe(true);

    //--------------------------------------------------
    // parse input
    //--------------------------------------------------

    if (fs::is_directory(path))
    {
        cv::glob(
            path.string() + "/*.jpg",
            imagePathList
        );
    }
    else if (fs::exists(path))
    {
        std::string suffix =
            path.extension().string();

        if (suffix == ".jpg" ||
            suffix == ".jpeg" ||
            suffix == ".png") {

            imagePathList.push_back(
                path.string()
            );
        }
        else if (
            suffix == ".mp4" ||
            suffix == ".avi" ||
            suffix == ".m4v" ||
            suffix == ".mpeg" ||
            suffix == ".mov" ||
            suffix == ".mkv") {

            isVideo = true;
        }
        else {

            printf(
                "suffix %s is wrong !!!\n",
                suffix.c_str()
            );

            delete yolov8;

            std::abort();
        }
    }
    else {

        printf(
            "input path does not exist: %s\n",
            path.string().c_str()
        );

        delete yolov8;

        return -1;
    }

    //--------------------------------------------------
    // infer config
    //--------------------------------------------------

    InferConfig config;

    config.score_thres =
        trt_config.score_thres;

    config.iou_thres =
        trt_config.iou_thres;

    //--------------------------------------------------
    // runtime buffer
    //--------------------------------------------------

    cv::Mat res;

    cv::Mat image;

    cv::Size size =
        cv::Size{ 640, 640 };

    int topk = 100;

    std::vector<Object> objs;

    cv::namedWindow(
        "result",
        cv::WINDOW_AUTOSIZE
    );

    //--------------------------------------------------
    // video infer
    //--------------------------------------------------

    if (isVideo)
    {
        cv::VideoCapture cap(
            path.string()
        );

        if (!cap.isOpened())
        {
            printf(
                "can not open %s\n",
                path.string().c_str()
            );

            delete yolov8;

            return -1;
        }

        while (cap.read(image))
        {
            objs.clear();

            yolov8->copy_from_Mat(
                image,
                size
            );

            auto start =
                std::chrono::system_clock::now();

            yolov8->infer();

            auto end =
                std::chrono::system_clock::now();

            yolov8->postprocess(
                objs,
                config.score_thres,
                config.iou_thres,
                topk
            );

            yolov8->draw_objects(
                image,
                res,
                objs,
                class_table.names,
                class_table.colors
            );

            auto tc =
                static_cast<double>(
                    std::chrono::duration_cast<
                    std::chrono::microseconds
                    >(end - start).count()
                    ) / 1000.0;

            printf(
                "cost %2.4lf ms\n",
                tc
            );

            cv::imshow(
                "result",
                res
            );

            if (cv::waitKey(10) == 'q')
            {
                break;
            }
        }
    }

    //--------------------------------------------------
    // image infer
    //--------------------------------------------------

    else
    {
        for (auto& p : imagePathList)
        {
            objs.clear();

            image =
                cv::imread(p);

            if (image.empty())
            {
                printf(
                    "can not read image: %s\n",
                    p.c_str()
                );

                continue;
            }

            yolov8->copy_from_Mat(
                image,
                size
            );

            auto start =
                std::chrono::system_clock::now();

            yolov8->infer();

            auto end =
                std::chrono::system_clock::now();

            yolov8->postprocess(
                objs,
                config.score_thres,
                config.iou_thres,
                topk
            );

            yolov8->draw_objects(
                image,
                res,
                objs,
                class_table.names,
                class_table.colors
            );

            auto tc =
                static_cast<double>(
                    std::chrono::duration_cast<
                    std::chrono::microseconds
                    >(end - start).count()
                    ) / 1000.0;

            printf(
                "cost %2.4lf ms\n",
                tc
            );

            cv::imshow(
                "result",
                res
            );

            cv::waitKey(0);
        }
    }

    //--------------------------------------------------
    // release
    //--------------------------------------------------

    cv::destroyAllWindows();

    delete yolov8;

    return 0;
}