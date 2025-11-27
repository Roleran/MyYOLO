
# 依赖库

```sh
# onnx 版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -zxvf onnxruntime-linux-x64-1.16.3.tgz
mv onnxruntime-linux-x64-1.16.3 onnxruntime

opencv
```


# 部署

C++ 程序，只要 #include "YoloV8.h" 并链接 libmyyolo.so，就能拥有一行代码调用 YOLO 的能力。
解耦: 主程序只负责读取视频和画图，复杂的数学运算和 ONNX 操作全被封装在库里了。


# 验证

./build/yolo_demo
Usage: ./yolo_demo <image_path/video_path/camera_index>
Example: ./yolo_demo ./test.jpg
Example: ./yolo_demo ./video.mp4
