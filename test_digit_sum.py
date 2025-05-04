import cv2
import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
from torchvision import transforms


# ...（模型定义和其他初始化代码保持不变）
class GestureCNNV2(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 10)),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(128 * 6 * 10, 512),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
# 修改MediaPipe初始化，允许检测两只手
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 检测两只手
    min_detection_confidence=0.5
)
# 数据预处理
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model_path = 'final_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureCNNV2().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
cap = cv2.VideoCapture(0)
padding = 20  # 边界框扩展像素
target_size = (64, 64)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("无法读取摄像头画面！")
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 初始化左右手预测值
    left_pred, right_pred = 0, 0
    # 初始化左右手边界框坐标
    left_bbox, right_bbox = None, None

    if results.multi_hand_landmarks:
        # 遍历所有检测到的手，并与 handedness 匹配
        for hand_idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)):
            # 获取左右手标签（MediaPipe返回的 handedness 是分类对象）
            hand_type = handedness.classification[0].label  # 'Left' 或 'Right'

            # 提取关键点坐标
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            # 计算边界框
            x_min = max(0, int(min(x_coords)) - padding)
            y_min = max(0, int(min(y_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_max = min(h, int(max(y_coords)) + padding)

            # 存储边界框坐标
            if hand_type == 'Left':
                left_bbox = (x_min, y_min, x_max, y_max)
            else:
                right_bbox = (x_min, y_min, x_max, y_max)

            # 裁剪和预处理
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue
            hand_roi = cv2.resize(hand_roi, target_size)
            hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)

            # 预测
            input_tensor = val_transform(hand_roi_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            # 根据左右手存储预测结果
            if hand_type == 'Left':
                left_pred = prediction

            else:
                right_pred = prediction

            print(f'左手数字为:{left_pred},右手数字为:{right_pred},左手加右手：{left_pred+right_pred}')


    # 绘制左手边框（绿色）
    if left_bbox is not None:
        x_min, y_min, x_max, y_max = left_bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Left: {left_pred}",
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # 绘制右手边框（红色）
    if right_bbox is not None:
        x_min, y_min, x_max, y_max = right_bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, f"Right: {right_pred}",
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # 计算总和并显示
    total = left_pred + right_pred
    cv2.putText(frame, f"Sum: {total}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

# ...（释放资源代码保持不变）
cap.release()
cv2.destroyAllWindows()