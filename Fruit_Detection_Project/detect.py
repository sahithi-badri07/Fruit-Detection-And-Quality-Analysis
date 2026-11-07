from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

def detect_image():
    img = cv2.imread("test.jpg")

    results = model(img)
    output = results[0].plot()

    count = len(results[0].boxes)
    print("Total Fruits:", count)

    cv2.imshow("Image Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        results = model(frame)
        output = results[0].plot()

        count = len(results[0].boxes)

        cv2.putText(output, f"Count: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Webcam Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Image Detection")
    print("2. Webcam Detection")
    choice = input("Enter choice: ")

    if choice == "1":
        detect_image()
    else:
        detect_webcam()