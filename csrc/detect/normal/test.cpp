#include <Windows.h>

#include <iostream>

#include <opencv2/opencv.hpp>

#include "yolov8det.h"

//--------------------------------------------------
// dll function typedef
//--------------------------------------------------

typedef const char* (*VersionFn)();

typedef YoloHandle(*CreateFn)(
    const char* engine_path,
    const char* class_json_path
    );

typedef void (*DestroyFn)(
    YoloHandle handle
    );

typedef int (*DetectFn)(
    YoloHandle handle,

    const YoloImage* image,

    YoloBox* out_boxes,

    int max_boxes,

    float score_thres,

    float iou_thres
    );

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
// main
//--------------------------------------------------

int main()
{
    //--------------------------------------------------
    // config
    //--------------------------------------------------

    TRTDetectConfig trt_config;

    trt_config.engine_path =
        "E:/steve/resources/dep_models/20260523/best.engine";

    trt_config.class_json_path =
        "E:/steve/resources/dep_models/20260523/classes.json";

    trt_config.input_path =
        "E:/steve/resources/dep_models/20260523/test.jpg";

    trt_config.device_id = 0;

    trt_config.score_thres = 0.25f;

    trt_config.iou_thres = 0.65f;

    //--------------------------------------------------
    // load dll
    //--------------------------------------------------

    HMODULE dll =
        LoadLibraryA(
            "E:\\steve\\code\\YOLOv8-TensorRT-main\\csrc\\detect\\normal\\out\\build\\Release\\yolov8_lib.dll"
        );

    if (!dll)
    {
        std::cout
            << "load dll failed\n";

        return -1;
    }

    std::cout
        << "load dll success\n";

    //--------------------------------------------------
    // get version api
    //--------------------------------------------------

    auto version_fn =
        reinterpret_cast<VersionFn>(
            GetProcAddress(
                dll,
                "yolo_version"
            )
            );

    if (!version_fn)
    {
        std::cout
            << "get version fn failed\n";

        FreeLibrary(dll);

        return -2;
    }

    std::cout
        << "dll version : "
        << version_fn()
        << std::endl;

    //--------------------------------------------------
    // get create api
    //--------------------------------------------------

    auto create_fn =
        reinterpret_cast<CreateFn>(
            GetProcAddress(
                dll,
                "yolo_create"
            )
            );

    if (!create_fn)
    {
        std::cout
            << "get create fn failed\n";

        FreeLibrary(dll);

        return -3;
    }

    std::cout
        << "get create fn success\n";

    //--------------------------------------------------
    // get destroy api
    //--------------------------------------------------

    auto destroy_fn =
        reinterpret_cast<DestroyFn>(
            GetProcAddress(
                dll,
                "yolo_destroy"
            )
            );

    if (!destroy_fn)
    {
        std::cout
            << "get destroy fn failed\n";

        FreeLibrary(dll);

        return -4;
    }

    std::cout
        << "get destroy fn success\n";

    //--------------------------------------------------
    // get detect api
    //--------------------------------------------------

    auto detect_fn =
        reinterpret_cast<DetectFn>(
            GetProcAddress(
                dll,
                "yolo_detect"
            )
            );

    if (!detect_fn)
    {
        std::cout
            << "get detect fn failed\n";

        FreeLibrary(dll);

        return -5;
    }

    std::cout
        << "get detect fn success\n";

    //--------------------------------------------------
    // create runtime
    //--------------------------------------------------

    YoloHandle handle =
        create_fn(
            trt_config.engine_path.c_str(),
            trt_config.class_json_path.c_str()
        );

    if (!handle)
    {
        std::cout
            << "create runtime failed\n";

        FreeLibrary(dll);

        return -6;
    }

    std::cout
        << "create runtime success\n";

    //--------------------------------------------------
    // read image
    //--------------------------------------------------

    cv::Mat img =
        cv::imread(
            trt_config.input_path
        );

    if (img.empty())
    {
        std::cout
            << "read image failed\n";

        destroy_fn(handle);

        FreeLibrary(dll);

        return -7;
    }

    std::cout
        << "read image success\n";

    //--------------------------------------------------
    // build yolo image
    //--------------------------------------------------

    YoloImage image;

    image.data =
        img.data;

    image.width =
        img.cols;

    image.height =
        img.rows;

    image.stride =
        static_cast<int>(img.step);

    image.channels =
        img.channels();

    //--------------------------------------------------
    // output buffer
    //--------------------------------------------------

    constexpr int MAX_BOXES = 100;

    YoloBox boxes[MAX_BOXES];

    //--------------------------------------------------
    // detect
    //--------------------------------------------------

    int num_boxes =
        detect_fn(
            handle,

            &image,

            boxes,

            MAX_BOXES,

            trt_config.score_thres,

            trt_config.iou_thres
        );

    if (num_boxes < 0)
    {
        std::cout
            << "detect failed : "
            << num_boxes
            << std::endl;

        destroy_fn(handle);

        FreeLibrary(dll);

        return -8;
    }

    //--------------------------------------------------
    // print result
    //--------------------------------------------------

    std::cout
        << "detect success\n";

    std::cout
        << "num boxes : "
        << num_boxes
        << std::endl;

    for (int i = 0; i < num_boxes; ++i)
    {
        const auto& box =

            boxes[i];

        std::cout
            << "----------------------------------\n";

        std::cout
            << "class_id : "
            << box.class_id
            << std::endl;

        std::cout
            << "score : "
            << box.score
            << std::endl;

        std::cout
            << "bbox : ["
            << box.x1
            << ", "
            << box.y1
            << ", "
            << box.x2
            << ", "
            << box.y2
            << "]"
            << std::endl;
    }

    //--------------------------------------------------
    // destroy runtime
    //--------------------------------------------------

    destroy_fn(handle);

    std::cout
        << "destroy runtime success\n";

    //--------------------------------------------------
    // free dll
    //--------------------------------------------------

    FreeLibrary(dll);

    std::cout
        << "free dll success\n";

    return 0;
}