from ultralytics import YOLO

def main():
    # 1. 加载模型
    # yolov8n.pt 是预训练权重，会自动下载（如果本地没有）
    # 也可以加载 yolov8n.yaml 从头训练（不推荐）
    model = YOLO('yolov8n.pt')
    print(model.info())

    # 2. 开始训练
    # data: 指向刚才配置好的 data.yaml
    # epochs: 训练轮数，测试流程设为 50-100 即可，正式训练建议 300+
    # imgsz: 图片大小，默认 640
    # batch: 批次大小，显存够大可以设为 16 或 32，不行就设 -1 (自动)
    # device: 0 表示使用你的第一张 GPU (RTX 3060)
    # workers: 数据加载进程数，Windows下如果报错建议设为 0 或 1，正常设为 4-8
    results = model.train(
        data='data.yaml',
        epochs=20,
        imgsz=160, # 640
        device=0,
        batch=16,
        name='run' # 训练结果保存的文件夹名称
    )

    # 3. 验证
    print("训练完成！")

if __name__ == '__main__':
    # Windows 下使用多进程必须放在 if __name__ == '__main__': 之下
    main()