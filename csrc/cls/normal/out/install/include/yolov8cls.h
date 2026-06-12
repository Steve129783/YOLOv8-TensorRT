#pragma once

#include <cstdint>

#ifdef _WIN32
#ifdef YOLO_CLS_EXPORTS
#define YOLO_CLS_API __declspec(dllexport)
#else
#define YOLO_CLS_API __declspec(dllimport)
#endif
#else
#define YOLO_CLS_API
#endif

//--------------------------------------------------
// ABI-safe handle
//--------------------------------------------------

typedef void* YoloClsHandle;

//--------------------------------------------------
// input image
//
// semantic lock:
//      CPU
//      uint8
//      HWC
//      BGR
//--------------------------------------------------

struct YoloImage
{
    const uint8_t* data;

    int width;

    int height;

    int stride;

    int channels;
};

//--------------------------------------------------
// classification result
//--------------------------------------------------

struct YoloClsResult
{
    int class_id;

    float score;

    char class_name[128];
};

#ifdef __cplusplus
extern "C" {
#endif

    //--------------------------------------------------
    // version
    //--------------------------------------------------

    YOLO_CLS_API const char* yolo_cls_version();

    //--------------------------------------------------
    // create
    //--------------------------------------------------

    YOLO_CLS_API YoloClsHandle yolo_cls_create(
        const char* engine_path,
        const char* class_json_path
    );

    //--------------------------------------------------
    // destroy
    //--------------------------------------------------

    YOLO_CLS_API void yolo_cls_destroy(
        YoloClsHandle handle
    );

    //--------------------------------------------------
    // predict
    //
    // return:
    //      >0 : result count
    //       0 : no result / below threshold
    //      <0 : error
    //--------------------------------------------------

    YOLO_CLS_API int yolo_cls_predict(
        YoloClsHandle handle,
        const YoloImage* image,
        YoloClsResult* out_results,
        int max_results,
        float score_thres
    );

#ifdef __cplusplus
}
#endif