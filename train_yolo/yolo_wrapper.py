
import cv2, time
from ultralytics import YOLO

class MyYOLODetector:
    def __init__(self, model_path, confidence=0.5):
        """
        初始化检测器
        :param model_path: 训练好的 .pt 文件路径
        :param confidence: 置信度阈值，低于这个分数的框会被忽略
        """
        print(f"正在加载模型: {model_path} ...")
        self.model = YOLO(model_path)
        self.conf = confidence
        print("模型加载完成！")

    def detect_image(self, image_path, show=True):
        """
        对单张图片进行检测
        """
        # 1. 执行推理
        # save=True 会把结果保存到 runs/detect/predict 文件夹
        results = self.model.predict(source=image_path, conf=self.conf, save=True)
        
        # 2. 获取第一张图的结果 (因为我们可以一次传多张，这里只取第一张)
        result = results[0]

        # 3. 打印检测到的数量
        print(f"检测到 {len(result.boxes)} 个目标")
        
        # 4. 如果需要弹窗显示
        if show:
            # result.plot() 会返回一个画好框的 numpy 数组 (BGR格式)
            annotated_frame = result.plot()
            
            # 使用 OpenCV 显示
            cv2.imshow("MyYOLO Result", annotated_frame)
            print("按任意键退出显示窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def detect_video(self, video_path):
        """
        [新增] 视频流检测功能
        """
        # 1. 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查是否打开成功
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return

        print(f"开始播放视频: {video_path}")
        print("提示: 按键盘上的 'q' 键退出播放")

        # 2. 循环读取帧
        while True:
            start_time = time.time() # 记录开始时间用于算FPS
            
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束")
                break

            # 3. 送入模型推理
            # verbose=False 让控制台不要一直刷屏打印信息
            results = self.model.predict(source=frame, conf=self.conf, verbose=False)
            result = results[0]

            # 4. 绘制检测框
            # result.plot() 返回的是画好框的 BGR 图片，可以直接给 OpenCV 用
            annotated_frame = result.plot()

            # 5. 计算并显示 FPS (帧率)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            # 在画面左上角写上 FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 6. 显示画面
            cv2.imshow("MyYOLO Video Detection", annotated_frame)

            # 7. 退出检测逻辑 (按 'q' 退出)
            # waitKey(1) 表示等待1毫秒，如果设为0则会暂停等待按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 8. 释放资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- 这里是调用示例 ---
    
    # 1. 指向你刚才训练出的最佳权重文件
    # 注意：路径可能需要根据你的实际文件夹调整，通常在 runs/detect/run/weights/best.pt
    # model_file = "runs/detect/run/weights/best.pt"
    model_file = "yolov8n.pt"
    
    # 2. 实例化你的类
    # 试着把 confidence 设低一点（比如 0.2），看看是不是框变多了？
    detector = MyYOLODetector(model_file, confidence=0.5)
    
    # 3. 找一张图来测试
    # 最好找一张你放在 datasets/images 里的图
    test_img = "datasets/images/1_41.png"


    # 视频路径 (请修改为你电脑上实际的视频路径)
    video_file = "driving.mp4" 
    
    # detector.detect_image(test_img)
    detector.detect_video(video_file)