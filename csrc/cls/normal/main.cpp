//
// Created by ubuntu on 4/27/24.
//
#include "opencv2/opencv.hpp"
#include "yolov8-cls.hpp"
#include <chrono>
#include <nlohmann/json.hpp>

namespace fs = ghc::filesystem;
//--------------------------------------------------
// runtime config
//--------------------------------------------------

struct TRTClsConfig
{
    std::string engine_path;

    std::string class_json_path;

    std::string input_path;

    int device_id = 0;

    float score_thres = 0.25f;
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

int main(int argc, char** argv)
{
//--------------------------------------------------
// config
//--------------------------------------------------

    TRTClsConfig trt_config;

    trt_config.engine_path =
        "E:/steve/resources/dep_models/20260611/best.engine";

    trt_config.class_json_path =
        "E:/steve/resources/dep_models/20260611/classes.json";

    trt_config.input_path =
        "E:/steve/resources/dep_models/20260611/test.jpg";

    trt_config.device_id = 0;

    trt_config.score_thres = 0.1f;

    // cuda:0
    cudaSetDevice(trt_config.device_id);
    fs::path path(trt_config.input_path);
    std::vector<std::string> imagePathList;
    bool                     isVideo{false};
    ClassTable table = load_class_info(trt_config.class_json_path);

    auto yolov8_cls = new YOLOv8_cls(trt_config.engine_path);
    yolov8_cls->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension().string();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path.string());
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

    cv::Mat  res, image;
    cv::Size size = cv::Size{224, 224};

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path.string());

        if (!cap.isOpened()) {
            fprintf(
                stderr,
                "can not open %s\n",
                path.string().c_str()
            );
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8_cls->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_cls->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_cls->postprocess(objs);
            yolov8_cls->draw_objects(image, res, objs, table.names);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_cls->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_cls->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_cls->postprocess(objs);
            yolov8_cls->draw_objects(image, res, objs, table.names);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8_cls;
    return 0;
}
