import torch
from dataset import create_datasets, create_loaders
from model import GestureCNNV2
from tqdm import tqdm

# 训练配置
CFG = {
    "batch_size": 4,
    "lr": 0.001,
    "epochs": 30,
    "save_path": "final_model.pth",
    "device": "cuda"
}


def main():
    # 初始化数据
    train_set, test_set = create_datasets()
    train_loader, test_loader = create_loaders(train_set, test_set, CFG["batch_size"])

    # 初始化模型
    model = GestureCNNV2().to(CFG["device"])
    criterion = torch.nn.CrossEntropyLoss().to(CFG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4,)
    # 训练循环
    best_train_acc = 0.0
    for epoch in range(CFG["epochs"]):
        model.train()
        total_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}",unit="batch"):
            inputs = inputs.to(CFG["device"])
            labels = labels.to(CFG["device"])

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_set)

        # 保存最佳模型（基于训练准确率）
        if train_acc > best_train_acc:
            torch.save(model.state_dict(), CFG["save_path"])
            best_train_acc = train_acc
            print(f"Epoch {epoch + 1}: New best model saved! Train Acc: {train_acc:.2%}")

        # 打印日志
        print(f"Epoch {epoch + 1}/{CFG['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Acc: {train_acc:.2%}")

    # 最终测试集评估
    model.load_state_dict(torch.load(CFG["save_path"]))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(CFG["device"])
            labels = labels.to(CFG["device"])
            test_correct += (model(inputs).argmax(1) == labels).sum().item()

    test_acc = test_correct / len(test_set)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()