r"""AI 编写的对照学习版；原始实现保留在 阶段1.py。

建议先读 main() 理解流程，再读各函数的输入、输出和状态修改。
运行：.\.venv\Scripts\python.exe .\阶段1重构.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


class CNN(nn.Module):
    """只定义模型计算；不负责读取数据、训练或保存。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.net(images)


def build_dataloaders(data_dir: Path, batch_size: int, split_seed: int):
    """构建固定划分的数据加载器；调用此函数才会读取或下载数据。"""
    CIFAR10.url = (
        "https://hf-mirror.com/datasets/MIT-OL-AI-D/"
        "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
    )
    # 对这里的 RGB 图片，保持原来的转换含义；浮点运算顺序可能带来微小舍入差异。
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    # 划分与采样各用一个随机状态，避免修改划分代码时顺带改变采样顺序。
    split_generator = torch.Generator().manual_seed(split_seed)
    shuffle_generator = torch.Generator().manual_seed(split_seed)
    train_data, val_data = random_split(dataset, [45000, 5000], generator=split_generator)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=shuffle_generator,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """原地更新传入模型，返回本轮按样本平均的训练 loss。"""
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # criterion 使用默认 reduction='mean'；末批不足时也按样本正确统计。
        count = targets.size(0)
        loss_sum += loss.item() * count
        sample_count += count

    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """切换为验证模式，只计算指标；返回平均 loss 和 [0, 1] 范围的准确率。"""
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        count = targets.size(0)
        loss_sum += loss.item() * count
        correct += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += count

    return loss_sum / sample_count, correct / sample_count


@torch.no_grad()
def verify_checkpoint(model, checkpoint_path: Path, images, device) -> float:
    """用新模型加载文件，比较同一批输入的 logits；不恢复训练状态。"""
    model.eval()
    expected = model(images.to(device))

    restored_model = CNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_model.eval()
    actual = restored_model(images.to(device))

    # 让检查失败时明确报错，避免只打印差异却被误当作“验证通过”。
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return (actual - expected).abs().max().item()


def main():
    # 配置集中在入口；先保持原实验的模型、轮数、batch 和优化设置。
    num_epochs, batch_size, lr = 5, 8, 0.001
    seed = 1
    # 同时固定模型初始化；这仍不承诺不同硬件/版本间逐位一致。
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    checkpoint_dir = project_dir / "checkpoints" / "阶段1重构"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(data_dir, batch_size, seed)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"device={device}, train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{num_epochs}, train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2%}"
        )

        checkpoint_path = checkpoint_dir / f"ckpt_ep{epoch}.pt"
        # 保存行为留在入口，让轮次与文件的对应关系一眼可见。
        # 这保留原版的三个字段，不声称已经支持完整断点续训。
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    sample_images, _ = next(iter(val_loader))
    max_difference = verify_checkpoint(model, checkpoint_path, sample_images, device)
    print(f"最后一轮 checkpoint：{checkpoint_path}")
    print(f"保存加载检查通过，固定验证 batch 的 logits 最大差异：{max_difference:.8f}")


if __name__ == "__main__":
    main()
