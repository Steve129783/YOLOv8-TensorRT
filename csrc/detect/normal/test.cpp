#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef HMODULE DllHandle;
#else
#include <dlfcn.h>
typedef void* DllHandle;
#endif

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
    // platform detect message
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)
    std::cout << "[INFO] Running on Windows" << std::endl;
#else
    std::cout << "[INFO] Running on Linux" << std::endl;
#endif

    //--------------------------------------------------
    // config
    //--------------------------------------------------

    TRTDetectConfig trt_config;

#if defined(_WIN32) || defined(_WIN64)

    trt_config.engine_path =
        "E:/steve/resources/dep_models/20260523/best.engine";

    trt_config.class_json_path =
        "E:/steve/resources/dep_models/20260523/classes.json";

    trt_config.input_path =
        "E:/steve/resources/dep_models/20260523/test.jpg";

    const char* dll_path =
        "E:\\steve\\code\\YOLOv8-TensorRT-main\\csrc\\detect\\normal\\out\\build\\Release\\yolov8_lib.dll";

#else

    trt_config.engine_path =
        "/media/fast/dep_models/20260523/best.engine";

    trt_config.class_json_path =
        "/media/fast/dep_models/20260523/classes.json";

    trt_config.input_path =
        "/media/fast/dep_models/20260523/test.jpg";

    const char* dll_path =
        "./libyolov8_lib.so";

#endif

    //--------------------------------------------------
    // load dll / so
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)

    DllHandle dll =
        LoadLibraryA(
            dll_path
        );

#else

    DllHandle dll =
        dlopen(
            dll_path,
            RTLD_NOW
        );

#endif

    if (!dll)
    {
        std::cout
            << "load dll failed : "
            << dll_path
            << std::endl;

#if defined(_WIN32) || defined(_WIN64)
        std::cout
            << "Windows error code : "
            << GetLastError()
            << std::endl;
#else
        std::cerr
            << dlerror()
            << std::endl;
#endif

        return -1;
    }

    std::cout
        << "load dll success\n";

    //--------------------------------------------------
    // get version api
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)

    auto version_fn =
        reinterpret_cast<VersionFn>(
            GetProcAddress(
                dll,
                "yolo_version"
            )
            );

#else

    auto version_fn =
        reinterpret_cast<VersionFn>(
            dlsym(
                dll,
                "yolo_version"
            )
            );

#endif

    if (!version_fn)
    {
        std::cout
            << "get version fn failed\n";

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -2;
    }

    std::cout
        << "dll version : "
        << version_fn()
        << std::endl;

    //--------------------------------------------------
    // get create api
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)

    auto create_fn =
        reinterpret_cast<CreateFn>(
            GetProcAddress(
                dll,
                "yolo_create"
            )
            );

#else

    auto create_fn =
        reinterpret_cast<CreateFn>(
            dlsym(
                dll,
                "yolo_create"
            )
            );

#endif

    if (!create_fn)
    {
        std::cout
            << "get create fn failed\n";

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -3;
    }

    //--------------------------------------------------
    // get destroy api
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)

    auto destroy_fn =
        reinterpret_cast<DestroyFn>(
            GetProcAddress(
                dll,
                "yolo_destroy"
            )
            );

#else

    auto destroy_fn =
        reinterpret_cast<DestroyFn>(
            dlsym(
                dll,
                "yolo_destroy"
            )
            );

#endif

    if (!destroy_fn)
    {
        std::cout
            << "get destroy fn failed\n";

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -4;
    }

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

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -5;
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
            << "load image failed : "
            << trt_config.input_path
            << std::endl;

        destroy_fn(handle);

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -6;
    }

    std::cout
        << "image loaded\n";

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
    // get detect api
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)

    auto detect_fn =
        reinterpret_cast<DetectFn>(
            GetProcAddress(
                dll,
                "yolo_detect"
            )
            );

#else

    auto detect_fn =
        reinterpret_cast<DetectFn>(
            dlsym(
                dll,
                "yolo_detect"
            )
            );

#endif

    if (!detect_fn)
    {
        std::cout
            << "get detect fn failed\n";

        destroy_fn(handle);

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

        return -7;
    }

    //--------------------------------------------------
    // runtime detect
    //--------------------------------------------------

    constexpr int MAX_BOXES = 100;

    YoloBox boxes[MAX_BOXES];

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

#if defined(_WIN32) || defined(_WIN64)
        FreeLibrary(dll);
#else
        dlclose(dll);
#endif

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
    // free dll / so
    //--------------------------------------------------

#if defined(_WIN32) || defined(_WIN64)
    FreeLibrary(dll);
#else
    dlclose(dll);
#endif

    std::cout
        << "free dll success\n";

    return 0;
}