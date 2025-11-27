#include "../include/YoloV8.h"
#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API
#include <iostream>

// 初始化类别名称 (COCO 80类)
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// 构造函数
YoloV8::YoloV8(const std::string& modelPath, bool useGPU) {
    classNames = COCO_CLASSES;

    // 1. 初始化环境
    Ort::Env* ortEnv = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YoloV8");
    this->env = (void*)ortEnv;

    // 2. 配置 Session 选项
    Ort::SessionOptions* ortOptions = new Ort::SessionOptions();
    ortOptions->SetIntraOpNumThreads(1);
    ortOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // (如果要在虚拟机里尝试 CUDA，需要取消注释并确保 CUDA 库可用)
    // if (useGPU) {
    //      OrtSessionOptionsAppendExecutionProvider_CUDA(*ortOptions, 0);
    // }

    this->sessionOptions = (void*)ortOptions;

    // 3. 加载模型
    try {
        Ort::Session* ortSession = new Ort::Session(*ortEnv, modelPath.c_str(), *ortOptions);
        this->session = (void*)ortSession;
        std::cout << "ONNX Model loaded successfully: " << modelPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        exit(1);
    }
}

// 辅助：Letterbox 预处理 (保持长宽比缩放)
cv::Mat YoloV8::formatToSquare(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

// 推理核心
std::vector<ObjectDetection> YoloV8::detect(const cv::Mat& image, float confThreshold, float nmsThreshold) {
    std::vector<ObjectDetection> results;

    // --- 1. 预处理 ---
    cv::Mat modelInput = formatToSquare(image);
    cv::Mat blob;
    // YOLOv8 要求归一化到 0-1 (scale=1/255.0)
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);

    // --- 2. 准备 ONNX 输入 ---
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // blob 已经是 NCHW 格式，且是 float32
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    size_t inputTensorSize = 1 * 3 * inputHeight * inputWidth;
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)blob.data, inputTensorSize, inputShape.data(), inputShape.size()
    );

    // 获取输入输出节点名称 (YOLOv8 通常输入是 "images", 输出是 "output0")
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    // --- 3. 运行推理 ---
    Ort::Session* ortSession = (Ort::Session*)this->session;
    auto outputTensors = ortSession->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1
    );

    // --- 4. 解析输出 ---
    // 输出形状: [1, 84, 8400] (84 = 4个坐标 + 80个类别, 8400是anchors)
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    
    // 转置处理：我们需要把 [1, 84, 8400] 理解为 [8400, 84] 以便遍历
    // 这里的 outputData 是一维数组，我们需要根据索引跳跃来读取
    
    int dimensions = 84; // 4 + 80
    int rows = 8400;     // anchors 数量

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 计算缩放比例 (用来把 640x640 的框还原回原图)
    float x_factor = modelInput.cols / (float)inputWidth; // 这里其实是1，因为我们做了square padding
    float y_factor = modelInput.rows / (float)inputHeight; 

    // 遍历所有 8400 个 anchor
    for (int i = 0; i < rows; ++i) {
        // 找出得分最高的类别
        // 在内存中，第 i 个 anchor 的数据并不是连续的，因为是 [84, 8400] 格式
        // 坐标 x 在 outputData[0 * 8400 + i]
        // 坐标 y 在 outputData[1 * 8400 + i]
        // ...
        // 类别 0 在 outputData[4 * 8400 + i]
        
        float maxScore = 0.0f;
        int classId = -1;

        // 遍历 80 个类别，找到最大置信度
        for (int c = 0; c < 80; ++c) {
             float score = outputData[(4 + c) * rows + i];
             if (score > maxScore) {
                 maxScore = score;
                 classId = c;
             }
        }

        if (maxScore >= confThreshold) {
            // 获取坐标
            float x = outputData[0 * rows + i];
            float y = outputData[1 * rows + i];
            float w = outputData[2 * rows + i];
            float h = outputData[3 * rows + i];

            // YOLOv8 输出的是中心点 (cx, cy, w, h)，需要转为左上角 (x, y, w, h)
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxScore);
            class_ids.push_back(classId);
        }
    }

    // --- 5. NMS (非极大值抑制) ---
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    // --- 6. 封装结果 ---
    for (int idx : indices) {
        ObjectDetection obj;
        obj.box = boxes[idx];
        obj.confidence = confidences[idx];
        obj.class_id = class_ids[idx];
        obj.className = classNames[obj.class_id];
        results.push_back(obj);
    }

    return results;
}