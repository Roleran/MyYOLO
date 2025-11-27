#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 定义一个结构体来保存检测结果
struct ObjectDetection {
    int class_id;       // 类别ID (0=person, 1=bicycle, etc.)
    float confidence;   // 置信度
    cv::Rect box;       // 边框 (x, y, width, height)
    std::string className; // 类别名称
};

// YOLOv8 封装类
class YoloV8 {
public:
    // 构造函数：传入模型路径，是否使用GPU(虚拟机建议先False)
    YoloV8(const std::string& modelPath, bool useGPU = false);
    
    // 核心推理函数
    std::vector<ObjectDetection> detect(const cv::Mat& image, float confThreshold = 0.25, float nmsThreshold = 0.45);

private:
    // Pimpl 模式或直接隐藏实现细节，这里为了简单直接放成员变量
    // 但因为要引用 ONNX Runtime 的头文件，为了保持头文件纯净，
    // 我们将在 .cpp 中包含 onnxruntime 头文件，这里用 void* 指针隐藏 Session
    void* session; 
    void* env;
    void* sessionOptions;
    
    // 模型输入尺寸
    const int inputWidth = 640;
    const int inputHeight = 640;
    // 检测会更快
    // const int inputWidth = 320;
    // const int inputHeight = 320;
    
    // COCO 80类名称
    std::vector<std::string> classNames;
    
    // 内部辅助函数：预处理（Letterbox）
    cv::Mat formatToSquare(const cv::Mat& source);
};