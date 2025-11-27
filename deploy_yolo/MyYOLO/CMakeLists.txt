cmake_minimum_required(VERSION 3.10)
project(MyYOLO)

set(CMAKE_CXX_STANDARD 17)

# 1. 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 2. 设置 ONNX Runtime 路径 (指向你解压的文件夹)
set(ONNX_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)

# 3. 编译动态库 (libmyyolo.so)
add_library(myyolo SHARED src/YoloV8.cpp)
target_link_libraries(myyolo ${OpenCV_LIBS} onnxruntime)

# 4. 编译可执行程序 demo (main.cpp)
add_executable(yolo_demo examples/main.cpp)
# 链接我们的库和OpenCV
target_link_libraries(yolo_demo myyolo ${OpenCV_LIBS})

# 5. 设置 RPATH，确保运行程序时能找到 libonnxruntime.so
set_target_properties(yolo_demo PROPERTIES BUILD_RPATH "${ONNX_ROOT}/lib")