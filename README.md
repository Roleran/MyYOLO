
# 自训练数据集

labelimg

# pt 转 onnx

# 部署 yolo


# 优化策略

## “视频播放过快”

问题一：为什么 cv 直接播放视频速度非常快？
原因：
cv::VideoCapture 读取视频时，它是一个“解码器”。它只负责把视频帧一张张解压出来，它不管视频原本的时间戳（TimeStamp）或帧率（FPS）。
如果 CPU 解码速度很快（比如 1ms 解码一张），而没有人为加延时，它就会以几百 FPS 的速度狂播，看起来就像快进。
解决办法：
人为控制每一帧的停留时间。
在 main.cpp 的 while 循环中，修改 cv::waitKey(1) 的逻辑。


## “推理耗时太长(100ms)”

问题二：耗时 100ms，如何优化？
现状分析：
现在耗时 100ms（即 10 FPS），这导致加上检测后，视频会从“太快”变成“卡顿”（因为 30 FPS 的视频每秒只能处理 10 帧）。
主要原因是：在虚拟机里运行，且使用的是 CPU 推理。

以下是 4 种优化方案，按推荐程度排序：

方法 1：调整 ONNX Runtime 线程数 (最简单，可能有 20-30% 提升)
CPU 推理时，线程数不是越多越好，太多的线程会造成上下文切换开销。通常设置为 CPU 物理核心数 效果最好。
修改 src/YoloV8.cpp 中的 sessionOptions 配置：
// ortOptions->SetIntraOpNumThreads(1); 

// 修改为：
// 如果你分配给虚拟机 4 个核，就设为 4。不要设太大。
ortOptions->SetIntraOpNumThreads(4);
然后重新编译 make。

方法 2：降低输入分辨率 (牺牲一点精度，速度翻倍)
YOLOv8n 默认是 640x640。对于 CPU 来说计算量还是有点大。如果目标物体不是特别小（比如远处的蚂蚁），改成 320x320 或 416x416 速度会快非常多。
修改 include/YoloV8.h：
// const int inputWidth = 640;
// const int inputHeight = 640;

// 修改为 320 (速度通常提升 3-4 倍，从 100ms 降到 30ms 左右)
const int inputWidth = 320;
const int inputHeight = 320;
注意：修改后需要重新编译。

方法 3：跳帧检测 (工程技巧，保持视频流畅)
如果必须用 CPU 且无法降低耗时，为了让视频看起来流畅（30FPS），可以每隔几帧检测一次，中间的帧直接画上一帧的框（或者用光流法跟踪，但那个复杂了，先用简单的）。
修改 main.cpp 的逻辑：
```c++
int frameCounter = 0;
int detectInterval = 3; // 每 3 帧检测一次 (检测 1 帧，跳过 2 帧)
std::vector<ObjectDetection> lastObjects; // 保存上一次的结果

while (true) {
    cap >> frame;
    if (frame.empty()) break;

    frameCounter++;

    // 只有在特定帧才进行耗时的 detect
    if (frameCounter % detectInterval == 0) {
        lastObjects = detector.detect(frame, 0.4, 0.45);
    }

    // 每一帧都画图（如果是跳过的帧，就画上一帧的结果）
    draw_objects(frame, lastObjects);

    cv::imshow("YOLOv8 Live", frame);
    if (cv::waitKey(1) == 27) break; // 这里改回 1，因为处理本身已经很慢了
}
```
这样视频看起来会流畅很多，虽然检测框的刷新率只有 10 FPS，但背景画面是 30 FPS。

方法 4：解决根本问题 —— 使用 GPU (环境大改)
