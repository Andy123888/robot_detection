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

# TensorRT模型路径
MODEL_PATH = HOME / "robot_project/model/best.engine"

RESULT_DIR = HOME / "robot_project/results"
ERROR_DIR = RESULT_DIR / "errors"

CAMERA_ID = 0

# YOLO基础检测阈值
PREDICT_CONF = 0.25

BOTTLE_CONF = 0.50
MOUSE_CONF = 0.60

IMG_SIZE = 640

RESULT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# ROS2节点
class YoloNode(Node):

    def __init__(self):
        super().__init__("yolo_detection_node")
        self.publisher = self.create_publisher(String, "/yolo/detections", 10)
        self.get_logger().info("YOLO ROS2 node started.")

def best_confidence(detections, class_name):
    # Return the maximum confidence of the specified class in the current frame
    # return 0.0 if no detection is found.
    values = [d["confidence"] for d in detections if d["class"] == class_name]
    return max(values) if values else 0.0

def main():
    rclpy.init()
    node = YoloNode()

    print("正在加载模型：", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))
    print("\n模型类别：", model.names)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # USB Camera Exposure Settings
    os.system("v4l2-ctl -d /dev/video0 -c auto_exposure=1")
    os.system("v4l2-ctl -d /dev/video0 -c exposure_time_absolute=466")

    if not cap.isOpened():
        print("摄像头打开失败")
        node.destroy_node()
        rclpy.shutdown()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS)

    if camera_fps <= 1:
        camera_fps = 20.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = RESULT_DIR / (f"demo_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        camera_fps,
        (width, height)
    )

    # Detection Log
    log_path = RESULT_DIR / (f"detection_log_{timestamp}.csv")

    log_file = open(
        log_path,"w",newline="",encoding="utf-8-sig"
    )

    log_writer = csv.writer(log_file)

    log_writer.writerow(
        ["time", "fps", "class", "confidence", "x1", "y1", "x2", "y2"]
    )

    test_path = RESULT_DIR / (
        f"test20_{timestamp}.csv"
    )

    test_file = open(
        test_path,"w",newline="",encoding="utf-8-sig"
    )

    test_writer = csv.writer(test_file)

    test_writer.writerow([
        "test_number","expected_class","detected_class",
        "bottle_confidence","mouse_confidence","correct"
    ])

    test_count = 0
    correct_count = 0

    smooth_fps = 0.0

    bottle_frames = 0
    mouse_frames = 0
    empty_frames = 0

    STABLE_FRAMES = 5

    print()
    print("==============================")
    print("YOLO+ROS2实时检测已启动")
    print("==============================")
    print("Q : 退出")
    print("S : 保存正确案例")
    print("E : 保存错误案例")
    print("B : 记录一次 bottle 测试")
    print("M : 记录一次 mouse 测试")
    print("A : 记录一次 bottle + mouse 同时测试")
    print("N : 记录一次 none 空画面测试")
    print("==============================")
    print()

    try:

        while True:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头失败")
                break

            # 将当前摄像头画面送入YOLO进行目标检测
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=PREDICT_CONF,
                verbose=False
            )

            result = results[0]

            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v)for v in box.xyxy[0].tolist()]
                    class_name = str(model.names[class_id])

                     # 根据类别设置不同阈值，过滤低置信度误检
                    if class_name == "bottle" and confidence < BOTTLE_CONF:
                        continue
                    if class_name == "mouse" and confidence < MOUSE_CONF:
                        continue

                    detections.append({
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1,y1,x2,y2]
                    })
            
            # Added: Judge bottle / mouse simultaneously
            bottle_conf = best_confidence(detections, "bottle")
            mouse_conf = best_confidence(detections, "mouse")

            # 连续多帧检测到目标
            if bottle_conf > 0.0:
                bottle_frames += 1
            else:
                bottle_frames = 0

            if mouse_conf > 0.0:
                mouse_frames += 1
            else:
                mouse_frames = 0

            if bottle_conf == 0.0 and mouse_conf == 0.0:
                empty_frames += 1
            else:
                empty_frames = 0    

            bottle_detected = bottle_frames >= STABLE_FRAMES
            mouse_detected = mouse_frames >= STABLE_FRAMES
            empty_detected = empty_frames >= STABLE_FRAMES

            # 两个类别都稳定检测到时为同时出现
            both_detected = (
                bottle_detected
                and mouse_detected
            )

            detected_classes = []

            if bottle_detected:
                detected_classes.append("bottle")
            if mouse_detected:
                detected_classes.append("mouse")
            if both_detected:
                scene_state = "bottle+mouse"
            elif bottle_detected:
                scene_state = "bottle"
            elif mouse_detected:
                scene_state = "mouse"
            elif empty_detected:
                scene_state = "none"
            else:
                scene_state = "unstable"

            detected_class_text = scene_state
            
            # Calculate and smooth the FPS
            elapsed = (
                time.perf_counter()
                - start_time
            )
            fps_now = 1.0 / elapsed if elapsed > 0 else 0.0
            if smooth_fps == 0:
                smooth_fps = fps_now
            else:
                smooth_fps = (
                    0.90 * smooth_fps + 0.10 * fps_now
                )

            display = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]

                class_name = det["class"]
                confidence = det["confidence"]
                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )
                cv2.putText(
                    display,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                display,
                f"FPS: {smooth_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            accuracy = correct_count / test_count * 100 if test_count else 0

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

            empty_text = (
                "Empty: YES"
                if empty_detected
                else "Empty: NO"
            )

            scene_text = f"Scene: {scene_state.upper()}"

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

            cv2.putText(
                display,
                empty_text,
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                scene_text,
                (20, 225),
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
                "empty_detected": empty_detected,

                "scene_state": scene_state,

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

            rclpy.spin_once(node,timeout_sec=0)


            video_writer.write(display)

            cv2.imshow("Jetson YOLO ROS2 Detection",display)

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

                cv2.imwrite(str(save_path),display)
                print("已保存正确案例：",save_path)

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
            # N：none / empty 
            # --------------------------
            elif key in [
                ord("b"),ord("m"),ord("a"),ord("n")
            ]:
                
                if test_count >= 20:
                    print("20次测试已经完成，不再记录新的测试。")
                    continue

                test_count += 1

                if key == ord("b"):
                    expected_class = "bottle"
                    correct = (
                        bottle_detected
                        and not mouse_detected
                        and not empty_detected
                    )

                elif key == ord("m"):
                    expected_class = "mouse"
                    correct = (
                        mouse_detected
                        and not bottle_detected
                        and not empty_detected
                    )

                elif key == ord("a"):
                    expected_class = "bottle+mouse"
                    correct = both_detected
                
                else:
                    expected_class = "none"
                    correct = empty_detected

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
                    detected_class_text,
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
                    print("\n======================")
                    print("20个目标测试完成")
                    print(f"正确数量：{correct_count}/20")
                    print(f"正确率：{accuracy:.1f}%")
                    print("======================")

    finally:
        # 程序退出时释放摄像头、文件和ROS2资源
        cap.release()
        video_writer.release()
        log_file.close()
        test_file.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

        print("\n程序已退出")
        print("视频：", video_path)
        print("检测日志：", log_path)
        print("20次测试：", test_path)


if __name__ == "__main__":
    main()