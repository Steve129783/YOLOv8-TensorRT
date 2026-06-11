#include "yolov8cls.h"

#include "yolov8-cls.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

//--------------------------------------------------
// internal context
//--------------------------------------------------

struct YoloClsContext
{
    YOLOv8_cls* classifier = nullptr;

    std::vector<std::string> class_names;
};

//--------------------------------------------------
// version
//--------------------------------------------------

const char* yolo_cls_version()
{
    return "0.1.0";
}

//--------------------------------------------------
// create
//--------------------------------------------------

YoloClsHandle yolo_cls_create(
    const char* engine_path,
    const char* class_json_path
)
{
    if (engine_path == nullptr ||
        class_json_path == nullptr) {

        return nullptr;
    }

    try {

        auto* ctx =
            new YoloClsContext();

        ctx->classifier =
            new YOLOv8_cls(engine_path);

        //--------------------------------------------------
        // original repo:
        //
        // make_pipe():
        //      cudaMalloc
        //      cudaHostAlloc
        //      warmup
        //--------------------------------------------------

        ctx->classifier->make_pipe(true);

        //--------------------------------------------------
        // load class metadata
        //--------------------------------------------------

        std::ifstream ifs(class_json_path);

        if (!ifs.is_open()) {

            delete ctx->classifier;

            delete ctx;

            return nullptr;
        }

        nlohmann::json j;

        ifs >> j;

        if (!j.is_array()) {

            delete ctx->classifier;

            delete ctx;

            return nullptr;
        }

        for (const auto& item : j)
        {
            if (!item.contains("name")) {

                delete ctx->classifier;

                delete ctx;

                return nullptr;
            }

            ctx->class_names.push_back(
                item["name"].get<std::string>()
            );
        }

        if (ctx->class_names.empty()) {

            delete ctx->classifier;

            delete ctx;

            return nullptr;
        }

        return reinterpret_cast<YoloClsHandle>(
            ctx
            );
    }
    catch (...) {

        return nullptr;
    }
}

//--------------------------------------------------
// destroy
//--------------------------------------------------

void yolo_cls_destroy(
    YoloClsHandle handle
)
{
    if (handle == nullptr) {
        return;
    }

    auto* ctx =
        reinterpret_cast<YoloClsContext*>(handle);

    delete ctx->classifier;

    ctx->classifier = nullptr;

    delete ctx;
}

//--------------------------------------------------
// predict
//--------------------------------------------------

int yolo_cls_predict(
    YoloClsHandle handle,
    const YoloImage* image,
    YoloClsResult* result
)
{
    //--------------------------------------------------
    // validate
    //--------------------------------------------------

    if (handle == nullptr ||
        image == nullptr ||
        result == nullptr) {

        return -1;
    }

    if (image->data == nullptr) {

        return -2;
    }

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->stride <= 0 ||
        image->channels <= 0) {

        return -3;
    }

    //--------------------------------------------------
    // current semantic lock:
    //
    // CPU
    // uint8
    // HWC
    // BGR
    //--------------------------------------------------

    if (image->channels != 3) {

        return -4;
    }

    auto* ctx =
        reinterpret_cast<YoloClsContext*>(handle);

    if (ctx == nullptr ||
        ctx->classifier == nullptr ||
        ctx->class_names.empty()) {

        return -5;
    }

    try {

        //--------------------------------------------------
        // wrap external memory
        //--------------------------------------------------

        cv::Mat img(
            image->height,
            image->width,
            CV_8UC3,
            const_cast<uint8_t*>(image->data),
            static_cast<size_t>(image->stride)
        );

        //--------------------------------------------------
        // backend-native output
        //--------------------------------------------------

        std::vector<cls::Object> objs;

        //--------------------------------------------------
        // real operator order from cls main:
        //
        // YoloImage
        //     ¡ý
        // cv::Mat HWC BGR uint8
        //     ¡ý
        // copy_from_Mat(img)
        //     ¡ý
        // use engine input shape
        //     ¡ý
        // TensorRT infer()
        //     ¡ý
        // postprocess(objs)
        //     ¡ý
        // cls::Object[]
        //--------------------------------------------------

        ctx->classifier->copy_from_Mat(
            img
        );

        ctx->classifier->infer();

        ctx->classifier->postprocess(
            objs
        );

        if (objs.empty()) {

            return 0;
        }

        //--------------------------------------------------
        // filter + copy to ABI-safe output
        //--------------------------------------------------

const auto& obj =
    objs[0];

const int cls_id =
    static_cast<int>(
        obj.label
    );

result->class_id =
    cls_id;

result->score =
    static_cast<float>(
        obj.prob
    );

std::memset(
    result->class_name,
    0,
    sizeof(result->class_name)
);

if (cls_id >= 0 &&
    cls_id < static_cast<int>(
        ctx->class_names.size()
    ))
{
#ifdef _WIN32

    strncpy_s(
        result->class_name,
        sizeof(result->class_name),
        ctx->class_names[cls_id].c_str(),
        sizeof(result->class_name) - 1
    );

#else

    std::strncpy(
        result->class_name,
        ctx->class_names[cls_id].c_str(),
        sizeof(result->class_name) - 1
    );

#endif
}

return 1;
    }
    catch (...) {

        return -6;
    }
}