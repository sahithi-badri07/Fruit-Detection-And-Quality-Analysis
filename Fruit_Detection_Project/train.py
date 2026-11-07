from ultralytics import YOLO

def train_model():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train model on your dataset
    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640
    )

if __name__ == "__main__":
    train_model()