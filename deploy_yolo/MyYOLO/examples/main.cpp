#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../include/YoloV8.h" // 确保路径正确

// --- 辅助函数：绘制检测结果 ---
void draw_objects(cv::Mat& img, const std::vector<ObjectDetection>& objects) {
    for (const auto& obj : objects) {
        // 1. 画框
        cv::rectangle(img, obj.box, cv::Scalar(0, 255, 0), 2);

        // 2. 准备标签文字 (类别 + 置信度)
        std::string label = obj.className + " " + std::to_string(obj.confidence).substr(0, 4);

        // 3. 计算文字背景框大小
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        // 4. 画文字背景 (实心绿框)
        int top = std::max(obj.box.y, labelSize.height);
        cv::rectangle(img, 
                      cv::Point(obj.box.x, top - labelSize.height),
                      cv::Point(obj.box.x + labelSize.width, top + baseLine), 
                      cv::Scalar(0, 255, 0), cv::FILLED);
        
        // 5. 写字 (黑色字体)
        cv::putText(img, label, cv::Point(obj.box.x, top), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// --- 主函数 ---
int main(int argc, char** argv) {
    // 1. 检查参数
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path/video_path/camera_index>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./test.jpg" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./video.mp4" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];

    // 2. 初始化检测器
    // 假设模型文件在上一级目录的 models 文件夹中
    std::string modelPath = "../yolov8n.onnx"; 
    
    std::cout << "Loading model from: " << modelPath << " ..." << std::endl;
    YoloV8 detector(modelPath, false); // false = CPU 推理
    std::cout << "Model loaded." << std::endl;

    // 3. 尝试作为图片读取
    cv::Mat image = cv::imread(inputPath);

    if (!image.empty()) {
        // --- 图片处理模式 ---
        std::cout << "Processing Image: " << inputPath << std::endl;

        // 检测
        auto objects = detector.detect(image, 0.4, 0.4);
        
        // 绘图
        draw_objects(image, objects);

        // 显示并保存
        cv::imshow("YOLOv8 Detection", image);
        cv::imwrite("result.jpg", image); // 保存结果
        std::cout << "Result saved to result.jpg. Press any key to exit." << std::endl;
        cv::waitKey(0);

    } else {
        // --- 视频/摄像头处理模式 ---
        std::cout << "Input is not an image, trying as video..." << std::endl;
        
        cv::VideoCapture cap;
        
        // 检查输入是否为纯数字（例如 "0" 或 "1"），如果是则打开摄像头
        // std::all_of 检查字符串是否全为数字
        if (std::all_of(inputPath.begin(), inputPath.end(), ::isdigit)) {
            int devId = std::stoi(inputPath);
            cap.open(devId);
            std::cout << "Opening Camera " << devId << std::endl;
        } else {
            cap.open(inputPath);
            std::cout << "Opening Video " << inputPath << std::endl;
        }

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video or camera." << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Video ended." << std::endl;
                break;
            }

            // 计算FPS (可选)
            auto start = std::chrono::high_resolution_clock::now();

            // 检测
            auto objects = detector.detect(frame, 0.4, 0.45);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            float fps = 1000.0 / (duration.count() + 1e-5);

            // 绘图
            draw_objects(frame, objects);

            // 显示检测耗时
            cv::putText(frame, "Duration: " + std::to_string((int)duration.count()) + "ms", cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

            // 显示 FPS
            cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            cv::imshow("YOLOv8 Live", frame);

            // 按 ESC (ASCII 27) 退出
            if (cv::waitKey(1) == 27) break;
        }
        
        cap.release();
    }

    cv::destroyAllWindows();
    return 0;
}