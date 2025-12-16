
# FlyMorpher

## 目录
```
├── crop_yolo_dataset.py	# 数据集准备脚本
├── FlyMorpher.ipynb		# 分类模型训练
├── FlyDetector.ipynb		# 检测模型训练
├── PicFlyMorpher.ipynb		# 检测+分类推理
```

## 快速开始

1. 准备训练分类器所需的数据

    - 目录结构：

        ```
        project/
        ├── crop_yolo_dataset.py
        └── data/
          └── data-with-img/          # 原始数据
              ├── images/        # 放全套原始图片
              └── labels/        # 放同名的 YOLO txt 标签
        ```

    - 修改路径：

        ```python
        INPUT_DIR  = r"data/data-with-img"      # 原始数据
        OUTPUT_DIR = r"data/data-with-img-cls"  # 想输出的位置
        SPLIT_RATIO = 0.8                  # 80 % 训练，20 % 验证
        ```

    - 运行命令：

        ```bash
        python crop_yolo_dataset.py
        ```

    - 输出结果：

        ```
        data/data-with-img-cls/
        ├── train/
        │   ├── Ambiguous/
        │   ├── Long/
        │   └── Short/
        └── validation/
            ├── Ambiguous/
            ├── Long/
            └── Short/
        ```
    每个文件夹里都是 224×224 的裁剪图，可直接喂给分类模型。

2. 训练模型  
   - 检测器

     - 训练
       - 打开 `FlyDetector.ipynb` 
       - 设置使用的YOLO预训练网络版本（默认`yolov8n`） 
       - 设置数据集路径（原始数据，YOLO format with images格式） 
       - Run All  
     - 输出：
       - 生成检测器适用的数据集：`data/data-with-img-det`
       - 在`data/yolo_runs/data-with-img-det_train/weights`里输出`best.pt` 
   - 分类器

     - 训练

       - 打开 `FlyMorpher.ipynb` 
   
       - 设置训练集路径：
         ```python
           train_loader, classes = loadTrain("./data/data-with-img-cls", batchsize)
           val_loader, _ = loadTest("./data/data-with-img-cls", batchsize)
         ```
   
       - 设置权重输出路径
         ```python
         PATH = "path/to/tained/weights.pth"
         ```
     
       - Run All
     
     - 输出：在设置好的路径中生成权重
     
   
3. 运行推理


- 打开 `PicFlyMorpher.ipynb` 
- 在最下方修改
  - ```python
	  img = cv2.imread("path/to/img.jpg")
	  detector = Detector("path/to/yolo.pt", "path/to/classifier.pth")
    cv2.imwrite("path/to/result.jpg", process_img)
  
- Run All 

- 输出：在设定好的路径中生成结果图。

## 环境

```bash
pip install -r requirements.txt
```
