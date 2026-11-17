from ultralytics import YOLO

def main():
    model = YOLO("weights/best.pt")

    # Thay source bằng:
    # - Đường dẫn thư mục ảnh: "datasets/hard-hat/test/images"
    # - Đường dẫn video: "demo.mp4"
    # - Webcam trực tiếp: 0
    model.predict(
        source="datasets/hard-hat/test/images",
        conf=0.5,
        save=True,
        project="runs/detect",
        name="predict_exp",
        exist_ok=True
    )

if __name__ == "__main__":
    main()