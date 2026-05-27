#pragma once

#include <stdint.h>

#ifdef _WIN32
#define YOLO_API __declspec(dllexport)
#else
#define YOLO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    //--------------------------------------------------
    // opaque runtime handle
    //--------------------------------------------------

    typedef void* YoloHandle;

    //--------------------------------------------------
    // image input
    //
    // semantic:
    //
    // data:
    //      CPU uint8 pointer
    //
    // layout:
    //      HWC
    //
    // channel order:
    //      BGR
    //
    // dtype:
    //      uint8
    //
    // stride:
    //      bytes per row
    //--------------------------------------------------

    typedef struct YoloImage
    {
        const uint8_t* data;

        int width;

        int height;

        int stride;

        int channels;

    } YoloImage;

    //--------------------------------------------------
    // detection result
    //
    // bbox semantic:
    //
    // x1,y1:
    //      left-top
    //
    // x2,y2:
    //      right-bottom
    //
    // coordinate space:
    //      original image pixel space
    //--------------------------------------------------

    typedef struct YoloBox
    {
        float x1;

        float y1;

        float x2;

        float y2;

        float score;

        int class_id;

    } YoloBox;

    //--------------------------------------------------
    // version
    //--------------------------------------------------

    YOLO_API
        const char* yolo_version();

    //--------------------------------------------------
    // create runtime
    //
    // semantic:
    //
    // engine_path:
    //      TensorRT engine path
    //
    // class_json_path:
    //      class metadata json path
    //--------------------------------------------------

    YOLO_API
        YoloHandle yolo_create(
            const char* engine_path,
            const char* class_json_path
        );

    //--------------------------------------------------
    // destroy runtime
    //--------------------------------------------------

    YOLO_API
        void yolo_destroy(
            YoloHandle handle
        );

    //--------------------------------------------------
    // detect
    //
    // return:
    //      actual detected box count
    //--------------------------------------------------

    YOLO_API
        int yolo_detect(
            YoloHandle handle,

            const YoloImage* image,

            YoloBox* out_boxes,

            int max_boxes,

            float score_thres,

            float iou_thres
        );

#ifdef __cplusplus
}
#endif