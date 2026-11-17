import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    model.train(
        data="datasets/hard-hat/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="train_exp",
        exist_ok=True
    )

    Path("models").mkdir(exist_ok=True)
    best_pt = Path("runs/detect/train_exp/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, "models/best.pt")
        print("Đã lưu mô hình tốt nhất tại: models/best.pt")

if __name__ == "__main__":
    main()