import time
import cv2
from ultralytics import YOLO

# initial
MODEL_PATH = "best.engine"
CAMERA_ID = 0
IMG_SIZE = 640

BOTTLE_CONF = 0.45
MOUSE_CONF = 0.55

CAM_WIDTH = 1280
CAM_HEIGHT = 720

def main():
    print("Loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Classes:", model.names)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"ERROR: cannot open camera {CAMERA_ID}")
        print("Run: ls /dev/video*")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    print("Camera opened. Press q or ESC to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read camera frame")
            break

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=min(BOTTLE_CONF, MOUSE_CONF),
            device=0,
            verbose=False
        )

        result = results[0]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[cls_id]

            if name == "bottle":
                threshold = BOTTLE_CONF
                color = (0, 255, 0)
            elif name == "mouse":
                threshold = MOUSE_CONF
                color = (0, 165, 255)
            else:
                continue

            if confidence < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{name} {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Jetson YOLO Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()
