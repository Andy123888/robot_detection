from pathlib import Path
import random
import shutil

# 项目根目录
ROOT = Path(r"D:\others\robot_detection")

# 原始图片和标签目录
IMG_DIR = ROOT / "raw" / "images"
LBL_DIR = ROOT / "raw" / "labels"

# 输出数据集目录
OUT_DIR = ROOT / "dataset"

random.seed(42)

# ========================================
# 1. 检查目录
# ========================================

if not IMG_DIR.exists():
    print("错误：找不到图片文件夹：")
    print(IMG_DIR)
    input("按回车退出...")
    exit()

if not LBL_DIR.exists():
    print("错误：找不到标签文件夹：")
    print(LBL_DIR)
    input("按回车退出...")
    exit()

# ========================================
# 2. 获取图片
# ========================================

# 不再分别搜索 *.jpg 和 *.JPG
# 直接读取所有文件，再统一判断后缀
image_files = [
    file for file in IMG_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

# 为了让文件顺序稳定
image_files = sorted(image_files)

print("找到图片总数：", len(image_files))

if len(image_files) == 0:
    print("错误：没有找到 JPG、JPEG 或 PNG 图片。")
    input("按回车退出...")
    exit()

# ========================================
# 3. 检查标签
# ========================================

valid_images = []
missing_labels = []

for img in image_files:

    label = LBL_DIR / f"{img.stem}.txt"

    if label.exists():
        valid_images.append(img)
    else:
        missing_labels.append(img.name)

print("有标签的图片：", len(valid_images))
print("缺少标签的图片：", len(missing_labels))

if missing_labels:

    print("\n以下图片缺少标签：")

    for name in missing_labels:
        print(name)

# 只使用存在标签的图片
image_files = valid_images

if len(image_files) == 0:
    print("没有可用于训练的数据。")
    input("按回车退出...")
    exit()

# ========================================
# 4. 随机打乱
# ========================================

random.shuffle(image_files)

n = len(image_files)

# 80% train
n_train = int(n * 0.8)

# 10% val
n_val = int(n * 0.1)

# 剩余约10% test
splits = {
    "train": image_files[:n_train],
    "val": image_files[n_train:n_train + n_val],
    "test": image_files[n_train + n_val:]
}

# ========================================
# 5. 创建并复制数据
# ========================================

for split, files in splits.items():

    img_out = OUT_DIR / "images" / split
    lbl_out = OUT_DIR / "labels" / split

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in files:

        # 复制图片
        shutil.copy2(
            img,
            img_out / img.name
        )

        # 复制对应标签
        label = LBL_DIR / f"{img.stem}.txt"

        shutil.copy2(
            label,
            lbl_out / label.name
        )

# ========================================
# 6. 输出结果
# ========================================

print("\n==============================")
print("数据划分完成")
print("==============================")

print("总有效图片：", n)
print("训练集 train：", len(splits["train"]))
print("验证集 val：", len(splits["val"]))
print("测试集 test：", len(splits["test"]))

print("\n数据集保存位置：")
print(OUT_DIR)

input("\n按回车退出...")