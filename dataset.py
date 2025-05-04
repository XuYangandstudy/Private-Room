import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def create_datasets():
    data_root = r"G:\Python_Project\Learn_pytorch\pythonProject9\Number_Get_V2\hand_dataset"

    # 训练集增强策略
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 将输入图像缩放到64*64
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（增强）
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整颜色
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    train_set = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform
    )

    test_set = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=test_transform
    )

    return train_set, test_set


def create_loaders(train_set, test_set, batch_size=4):
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=5,pin_memory=True)
    test_loader = DataLoader(test_set, batch_size, num_workers=5,pin_memory=True)
    return train_loader, test_loader