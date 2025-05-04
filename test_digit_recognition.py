import cv2
import torch
import torch.nn as nn
import mediapipe as mp

from torchvision.transforms import transforms




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


# 图片预处理
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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
padding = 20  # 边界框扩展像素
target_size = (64, 64)
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
    number=0
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
            hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            input_tensor = val_transform(hand_roi).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                predictions = torch.argmax(output, dim=1).item()
                print(f'手势数字为{predictions}')
                number=predictions
            # 显示裁剪后的手部区域
            cv2.imshow("Hand ROI", hand_roi)


            # 绘制边界框和标签
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # 显示摄像头画面
    cv2.putText(frame, f"number: {number}",
                (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow("Hand Data Collection", frame)

        # 退出条件：按ESC键
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
