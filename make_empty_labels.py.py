from pathlib import Path

IMG_DIR = Path(r"D:\others\robot_detection\new70\images")
LBL_DIR = Path(r"D:\others\robot_detection\new70\labels")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

created = 0

for img in IMG_DIR.iterdir():
    if img.suffix not in IMG_EXTS:
        continue

    label = LBL_DIR / f"{img.stem}.txt"

    if not label.exists():
        label.touch()
        print("创建空标签：", label.name)
        created += 1

print()
print("完成，创建空标签：", created)