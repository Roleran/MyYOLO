
from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO("yolov8n.pt")

# 2. 导出为 ONNX 格式
# dynamic=True: 允许输入图片尺寸变化（不强制 640x640），这在工程中很灵活
# opset: ONNX 版本，12 或 13 兼容性较好
success = model.export(format="onnx", dynamic=True, opset=12)

print(f"导出状态: {success}")