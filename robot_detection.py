import cv2
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO


HOME = Path.home()

MODEL_PATH = HOME / "robot_project/model/best.engine"

RESULT_DIR = HOME / "robot_project/results"
ERROR_DIR = RESULT_DIR / "errors"

CAMERA_ID = 0

CONF = 0.50
IMG_SIZE = 640

RESULT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

class YoloNode(Node):

    def __init__(self):

        super().__init__("yolo_detection_node")

        self.publisher = self.create_publisher(
            String,
            "/yolo/detections",
            10
        )

        self.get_logger().info(
            "YOLO ROS2 node started."
        )

def best_confidence(detections, class_name):
    # Return the maximum confidence of the specified class in the current frame
    # return 0.0 if no detection is found.
    values = [
        det["confidence"]
        for det in detections
        if det["class"] == class_name
    ]
    return max(values) if values else 0.0

def main():

    rclpy.init()

    node = YoloNode()

    print("正在加载模型：")
    print(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    print("\n模型类别：")
    print(model.names)

    cap = cv2.VideoCapture(CAMERA_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():

        print("摄像头打开失败")
        node.destroy_node()
        rclpy.shutdown()
        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    camera_fps = cap.get(cv2.CAP_PROP_FPS)

    if camera_fps <= 1:
        camera_fps = 20.0

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    video_path = RESULT_DIR / (
        f"demo_{timestamp}.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        camera_fps,
        (width, height)
    )

    # Detection Log
    log_path = RESULT_DIR / (
        f"detection_log_{timestamp}.csv"
    )

    log_file = open(
        log_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    log_writer = csv.writer(log_file)

    log_writer.writerow([
        "time",
        "fps",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    test_path = RESULT_DIR / (
        f"test20_{timestamp}.csv"
    )

    test_file = open(
        test_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    )

    test_writer = csv.writer(test_file)

    test_writer.writerow([
        "test_number",
        "expected_class",
        "detected_class",
        "confidence",
        "correct"
    ])

    test_count = 0
    correct_count = 0

    smooth_fps = 0.0

    print()
    print("==============================")
    print("YOLO + ROS2 实时检测已启动")
    print("==============================")
    print("Q : 退出")
    print("S : 保存正确案例")
    print("E : 保存错误案例")
    print("B : 记录一次 bottle 测试")
    print("M : 记录一次 mouse 测试")
    print("A : 记录一次 bottle + mouse 同时测试")
    print("==============================")
    print()

    try:

        while True:

            start_time = time.perf_counter()

            ret, frame = cap.read()

            if not ret:
                print("读取摄像头失败")
                break


            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF,
                verbose=False
            )

            result = results[0]

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    x1, y1, x2, y2 = [
                        int(v)
                        for v in box.xyxy[0].tolist()
                    ]

                    class_name = str(
                        model.names[class_id]
                    )

                    detections.append({
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    })
            
            # Added: Judge bottle / mouse simultaneously
            bottle_conf = best_confidence(detections, "bottle")
            mouse_conf = best_confidence(detections, "mouse")

            bottle_detected = bottle_conf > 0.0
            mouse_detected = mouse_conf > 0.0
            both_detected = bottle_detected and mouse_detected

            detected_classes = []
            if bottle_detected:
                detected_classes.append("bottle")
            if mouse_detected:
                detected_classes.append("mouse")

            detected_class_text = (
                "+".join(detected_classes)
                if detected_classes
                else "NOT_DETECTED"
            )

            

            # FPS
            elapsed = (
                time.perf_counter()
                - start_time
            )

            fps_now = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            if smooth_fps == 0:

                smooth_fps = fps_now

            else:

                smooth_fps = (
                    0.90 * smooth_fps
                    + 0.10 * fps_now
                )


            # YOLO自动画框
            display = result.plot()

            cv2.putText(
                display,
                f"FPS: {smooth_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            accuracy = (
                correct_count / test_count * 100
                if test_count > 0
                else 0
            )

            cv2.putText(
                display,
                f"Test: {test_count}/20  "
                f"Accuracy: {accuracy:.1f}%",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            bottle_text = (
                f"Bottle: YES ({bottle_conf:.2f})"
                if bottle_detected
                else "Bottle: NO"
            )
            mouse_text = (
                f"Mouse: YES ({mouse_conf:.2f})"
                if mouse_detected
                else "Mouse: NO"
            )
            both_text = "Both: YES" if both_detected else "Both: NO"

            cv2.putText(
                display,
                bottle_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                mouse_text,
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                both_text,
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            # ROS2 publication
            ros_data = {
                "fps": round(smooth_fps, 2),
                "bottle_detected": bottle_detected,
                "mouse_detected": mouse_detected,
                "both_detected": both_detected,
                "detected_classes": detected_classes,
                "detections": []
            }

            for det in detections:

                ros_data["detections"].append({
                    "class": det["class"],
                    "confidence": round(
                        det["confidence"],
                        3
                    ),
                    "bbox": det["bbox"]
                })

                log_writer.writerow([
                    datetime.now().isoformat(),
                    round(smooth_fps, 2),
                    det["class"],
                    round(
                        det["confidence"],
                        4
                    ),
                    *det["bbox"]
                ])

            msg = String()

            msg.data = json.dumps(
                ros_data,
                ensure_ascii=False
            )

            node.publisher.publish(msg)

            rclpy.spin_once(
                node,
                timeout_sec=0
            )


            video_writer.write(display)

            cv2.imshow(
                "Jetson YOLO ROS2 Detection",
                display
            )

            key = cv2.waitKey(1) & 0xFF

            # Q：Exit
            if key == ord("q"):

                break

            # --------------------------
            # S：Correct Case
            # --------------------------
            elif key == ord("s"):

                save_path = RESULT_DIR / (
                    datetime.now().strftime(
                        "success_%Y%m%d_%H%M%S.jpg"
                    )
                )

                cv2.imwrite(
                    str(save_path),
                    display
                )

                print(
                    "已保存正确案例：",
                    save_path
                )

            # --------------------------
            # E：Error Case
            # --------------------------
            elif key == ord("e"):

                save_path = ERROR_DIR / (
                    datetime.now().strftime(
                        "error_%Y%m%d_%H%M%S.jpg"
                    )
                )

                cv2.imwrite(
                    str(save_path),
                    display
                )

                print(
                    "已保存错误案例：",
                    save_path
                )

            # --------------------------
            # B：bottle test
            # M：mouse test
            # A: bottle + mouse
            # --------------------------
            elif key in [
                ord("b"),
                ord("m"),
                ord("a")
            ]:
                
                test_count += 1

                if key == ord("b"):
                    expected_class = "bottle"
                    correct = bottle_detected and not mouse_detected

                elif key == ord("m"):
                    expected_class = "mouse"
                    correct = mouse_detected and not bottle_detected

                else:
                    expected_class = "bottle+mouse"
                    correct = both_detected

                if correct:
                    correct_count += 1
                else:
                    error_path = (
                        ERROR_DIR /
                        f"test_error_{test_count:02d}.jpg"
                    )
                    cv2.imwrite(str(error_path), display)

                test_writer.writerow([
                    test_count,
                    expected_class,
                    detected_class_text,,
                    round(bottle_conf, 4),
                    round(mouse_conf, 4),
                    correct
                ])

                test_file.flush()

                accuracy = (
                    correct_count
                    / test_count
                    * 100
                )

                print(
                    f"测试 {test_count}: "
                    f"实际={expected_class}, "
                    f"识别={detected_class_text}, "
                    f"bottle_conf={bottle_conf:.3f}, "
                    f"mouse_conf={mouse_conf:.3f}, "
                    f"正确={correct}, "
                    f"当前正确率="
                    f"{accuracy:.1f}%"
                )

                if test_count == 20:

                    print()
                    print(
                        "======================"
                    )
                    print(
                        "20个目标测试完成"
                    )
                    print(
                        f"正确数量："
                        f"{correct_count}/20"
                    )
                    print(
                        f"正确率："
                        f"{accuracy:.1f}%"
                    )
                    print(
                        "======================"
                    )

    finally:

        cap.release()

        video_writer.release()

        log_file.close()

        test_file.close()

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()

        print()
        print("程序已退出")
        print("视频：", video_path)
        print("检测日志：", log_path)
        print("20次测试：", test_path)


if __name__ == "__main__":
    main()