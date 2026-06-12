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

YOLO_CLS_API int yolo_cls_predict(
    YoloClsHandle handle,
    const YoloImage* image,
    YoloClsResult* out_results,
    int max_results,
    float score_thres
)
{
    //--------------------------------------------------
    // validate
    //--------------------------------------------------

    if (handle == nullptr ||
        image == nullptr ||
        out_results == nullptr) {

        return -1;
    }

    if (max_results <= 0) {
        return -2;
    }

    if (image->data == nullptr) {
        return -3;
    }

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->stride <= 0 ||
        image->channels <= 0) {

        return -4;
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
        return -5;
    }

    auto* ctx =
        reinterpret_cast<YoloClsContext*>(handle);

    if (ctx == nullptr ||
        ctx->classifier == nullptr ||
        ctx->class_names.empty()) {

        return -6;
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
        // sort by probability, high -> low
        //--------------------------------------------------

        std::sort(
            objs.begin(),
            objs.end(),
            [](const cls::Object& a, const cls::Object& b)
            {
                return a.prob > b.prob;
            }
        );

        //--------------------------------------------------
        // filter + copy to ABI-safe output
        //--------------------------------------------------

        int count = 0;

        for (const auto& obj : objs) {

            if (count >= max_results) {
                break;
            }

            const float score =
                static_cast<float>(obj.prob);

            if (score < score_thres) {
                continue;
            }

            const int cls_id =
                static_cast<int>(obj.label);

            YoloClsResult& dst =
                out_results[count];

            dst.class_id =
                cls_id;

            dst.score =
                score;

            std::memset(
                dst.class_name,
                0,
                sizeof(dst.class_name)
            );

            if (cls_id >= 0 &&
                cls_id < static_cast<int>(
                    ctx->class_names.size()
                    )) {

#ifdef _WIN32

                strncpy_s(
                    dst.class_name,
                    sizeof(dst.class_name),
                    ctx->class_names[cls_id].c_str(),
                    sizeof(dst.class_name) - 1
                );

#else

                std::strncpy(
                    dst.class_name,
                    ctx->class_names[cls_id].c_str(),
                    sizeof(dst.class_name) - 1
                );

#endif
            }

            ++count;
        }

        return count;
    }
    catch (...) {

        return -7;
    }
}