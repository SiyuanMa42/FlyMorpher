import os
import cv2
from pathlib import Path
from tqdm import tqdm
import random


def create_output_structure(base_dir):
    """创建输出目录结构"""
    splits = ["train", "validation"]
    classes = ["Ambiguous", "Long", "Short"]

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
    print(f"✅ 已创建输出目录结构: {base_dir}")


def yolo_to_pixels(img_shape, x_center, y_center, width, height):
    """将YOLO格式的归一化坐标转换为像素坐标"""
    h, w = img_shape[:2]
    x_center_px = int(x_center * w)
    y_center_px = int(y_center * h)
    width_px = int(width * w)
    height_px = int(height * h)

    # 计算左上角和右下角坐标
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    # 确保坐标在图片范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2


def crop_and_resize(image_path, label_path, output_dir, split="train", target_size=(224, 224)):
    """处理单张图片及其标签"""
    if not os.path.exists(label_path):
        print(f"⚠️  标签文件不存在: {label_path}")
        return

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 读取标签
    with open(label_path, "r") as f:
        lines = f.readlines()

    # 处理每个目标
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            print(f"❌ 标签格式错误: {label_path}, 行: {line}")
            continue

        # 类别映射
        class_names = {0: "Ambiguous", 1: "Long", 2: "Short"}
        if class_id not in class_names:
            print(f"⚠️  未知类别ID: {class_id}")
            continue

        class_name = class_names[class_id]

        # 转换为像素坐标并裁剪
        x1, y1, x2, y2 = yolo_to_pixels(
            img.shape, x_center, y_center, width, height)

        # 检查裁剪区域是否有效
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️  无效裁剪区域: {image_path} - 类别 {class_name}")
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️  裁剪结果为空: {image_path}")
            continue

        # 缩放为224x224
        crop_resized = cv2.resize(
            crop, target_size, interpolation=cv2.INTER_AREA)

        # 生成输出文件名
        img_basename = Path(image_path).stem
        output_filename = f"{img_basename}_{idx}.jpg"
        output_path = os.path.join(
            output_dir, split, class_name, output_filename)

        # 保存裁剪后的图片
        cv2.imwrite(output_path, crop_resized)


def process_dataset(input_dir, output_dir, split_ratio=0.9, img_ext=".jpg", seed=None):
    """
    处理整个数据集

    参数:
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        split_ratio: 训练集比例 (默认为0.9)
        seed: 可选，整数种子以便结果可复现（默认为 None，不设置则每次随机）
        img_ext: 图片文件扩展名
    """
    # 创建输出目录结构
    create_output_structure(output_dir)

    # 输入路径
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(images_dir):
        print(f"❌ 图片目录不存在: {images_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"❌ 标签目录不存在: {labels_dir}")
        return

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp"))]

    if not image_files:
        print(f"⚠️  在 {images_dir} 中未找到图片文件")
        return

    print(f"📸 找到 {len(image_files)} 张图片")

    # 可选设置随机种子以便复现
    if seed is not None:
        random.seed(seed)

    # 随机打乱然后按比例切分（更直观、可复现）
    random.shuffle(image_files)
    train_count = int(len(image_files) * split_ratio)

    # 处理每张图片
    for idx, img_file in enumerate(tqdm(image_files, desc="处理图片")):
        split = "train" if idx < train_count else "validation"

        # 构建文件路径
        img_path = os.path.join(images_dir, img_file)
        label_file = Path(img_file).stem + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # 处理图片
        crop_and_resize(img_path, label_path, output_dir, split)

    print("✅ 处理完成！")


if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = r"data/1210-403"  # 输入数据集目录
    OUTPUT_DIR = r"data/1210-403-cls"  # 输出数据集目录
    SPLIT_RATIO = 0.8  # 80%训练集，20%验证集
    IMAGE_EXT = ".jpg"  # 图片格式

    # 执行处理
    process_dataset(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                    split_ratio=SPLIT_RATIO, img_ext=IMAGE_EXT)
