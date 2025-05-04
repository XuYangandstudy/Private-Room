import cv2
import mediapipe as mp
import os
import time

# 初始化MediaPipe手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# 数据集保存根目录
save_dir = "hand_dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头，可改为视频文件路径
current_label = 0  # 当前手势标签（0~9）
collected_count = 0  # 已收集的样本数量
target_size = (64, 64)  # 裁剪后的图像尺寸
padding = 20  # 边界框扩展像素

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法读取摄像头画面！")
        break

    # 镜像翻转（更符合自然视角）
    frame = cv2.flip(frame, 1)

    # 转换为RGB格式供MediaPipe处理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # 只处理检测到的第一只手
        hand_landmarks = results.multi_hand_landmarks[0]

        # 提取所有关键点的x,y坐标（相对于图像尺寸）
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # 计算手部边界框
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # 扩展边界框（避免裁剪过紧）
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # 裁剪手部区域
        hand_roi = frame[y_min:y_max, x_min:x_max]

        # 调整图像尺寸
        if hand_roi.size != 0:
            hand_roi = cv2.resize(hand_roi, target_size)

            # 显示裁剪后的手部区域
            cv2.imshow("Hand ROI", hand_roi)

            # 按下数字键保存当前手势（0~9）
            key = cv2.waitKey(1) & 0xFF
            if 48 <= key <= 57:  # 数字0~9的ASCII码
                current_label = key - 48

                # 创建标签子目录（如果不存在）
                label_dir = os.path.join(save_dir, str(current_label))
                os.makedirs(label_dir, exist_ok=True)

                # 生成唯一文件名（时间戳）
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(label_dir, f"{current_label}_{timestamp}.jpg")

                # 保存图片
                cv2.imwrite(save_path, hand_roi)
                collected_count += 1
                print(f"保存: {save_path} | 总样本数: {collected_count}")

            # 绘制边界框和标签
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示摄像头画面
    cv2.imshow("Hand Data Collection", frame)

    # 退出条件：按ESC键
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print(f"数据收集完成！图片保存在 {save_dir} 目录中。")