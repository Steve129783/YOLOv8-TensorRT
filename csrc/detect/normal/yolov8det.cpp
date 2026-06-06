#include "yolov8det.h"

#include "yolov8.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

//--------------------------------------------------
// version
//--------------------------------------------------

const char* yolo_version()
{
    return "0.1.0";
}

//--------------------------------------------------
// create
//--------------------------------------------------

YoloHandle yolo_create(
    const char* engine_path,
    const char* class_json_path
)
{
    if (engine_path == nullptr ||
        class_json_path == nullptr) {

        return nullptr;
    }

    try {

        auto* detector =
            new YOLOv8(engine_path);

        //--------------------------------------------------
        // original repo:
        //
        // make_pipe():
        //      cudaMalloc
        //      cudaHostAlloc
        //      warmup
        //--------------------------------------------------

        detector->make_pipe(true);

        //--------------------------------------------------
        // load class metadata
        //--------------------------------------------------

        std::ifstream ifs(class_json_path);

        if (!ifs.is_open()) {

            delete detector;

            return nullptr;
        }

        nlohmann::json j;

        ifs >> j;

        if (!j.is_array()) {

            delete detector;

            return nullptr;
        }

        for (const auto& item : j)
        {
            detector->class_names_.push_back(
                item["name"].get<std::string>()
            );

            auto color =
                item["color"];

            detector->class_colors_.push_back(
                cv::Scalar(
                    color[0].get<int>(),
                    color[1].get<int>(),
                    color[2].get<int>()
                )
            );
        }

        detector->num_labels_ =
            static_cast<int>(j.size());

        if (detector->num_labels_ <= 0) {

            delete detector;

            return nullptr;
        }

        std::cout
            << "num labels : "
            << detector->num_labels_
            << std::endl;

        return reinterpret_cast<YoloHandle>(
            detector
            );
    }
    catch (...) {

        return nullptr;
    }
}

//--------------------------------------------------
// destroy
//--------------------------------------------------

void yolo_destroy(
    YoloHandle handle
)
{
    if (handle == nullptr) {
        return;
    }

    auto* detector =
        reinterpret_cast<YOLOv8*>(handle);

    delete detector;
}

//--------------------------------------------------
// detect
//--------------------------------------------------

int yolo_detect(
    YoloHandle handle,

    const YoloImage* image,

    YoloBox* out_boxes,

    int max_boxes,

    float score_thres,

    float iou_thres
)
{
    //--------------------------------------------------
    // validate
    //--------------------------------------------------

    if (handle == nullptr ||
        image == nullptr ||
        out_boxes == nullptr) {

        return -1;
    }

    if (image->data == nullptr) {

        return -2;
    }

    if (image->width <= 0 ||
        image->height <= 0 ||
        image->stride <= 0 ||
        image->channels <= 0 ||
        max_boxes <= 0) {

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

    auto* detector =
        reinterpret_cast<YOLOv8*>(handle);

    if (detector->num_labels_ <= 0) {

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
        // backend-native structure
        //--------------------------------------------------

        std::vector<det::Object> objs;

        //--------------------------------------------------
        // real operator order:
        //
        // cv::Mat
        //     ¡ý
        // letterbox / HWC->NCHW / normalize / cudaMemcpy
        //     ¡ý
        // TensorRT enqueue
        //     ¡ý
        // decode + NMS
        //     ¡ý
        // det::Object[]
        //--------------------------------------------------

        detector->copy_from_Mat(img);

        detector->infer();

        detector->postprocess(
            objs,
            score_thres,
            iou_thres,
            max_boxes
        );

        //--------------------------------------------------
        // copy to ABI-safe structure
        //--------------------------------------------------

        const int n =
            std::min(
                static_cast<int>(objs.size()),
                max_boxes
            );

        for (int i = 0; i < n; ++i)
        {
            const det::Object& obj =
                objs[i];

            out_boxes[i].x1 =
                static_cast<float>(
                    obj.rect.x
                    );

            out_boxes[i].y1 =
                static_cast<float>(
                    obj.rect.y
                    );

            out_boxes[i].x2 =
                static_cast<float>(
                    obj.rect.x +
                    obj.rect.width
                    );

            out_boxes[i].y2 =
                static_cast<float>(
                    obj.rect.y +
                    obj.rect.height
                    );

            out_boxes[i].score =
                static_cast<float>(
                    obj.prob
                    );

            //--------------------------------------------------
            // class id
            //--------------------------------------------------

            const int cls_id =
                static_cast<int>(
                    obj.label
                    );

            out_boxes[i].class_id =
                cls_id;

            //--------------------------------------------------
            // color
            //--------------------------------------------------

            if (
                cls_id >= 0 &&
                cls_id < static_cast<int>(
                    detector->class_colors_.size()
                    )
                )
            {
                const auto& color =
                    detector->class_colors_[cls_id];

                out_boxes[i].color_b =
                    static_cast<uint8_t>(
                        color[0]
                        );

                out_boxes[i].color_g =
                    static_cast<uint8_t>(
                        color[1]
                        );

                out_boxes[i].color_r =
                    static_cast<uint8_t>(
                        color[2]
                        );
            }
            else
            {
                out_boxes[i].color_b = 0;
                out_boxes[i].color_g = 255;
                out_boxes[i].color_r = 0;
            }
        }

        return n;
    }
    catch (...) {

        return -6;
    }
}