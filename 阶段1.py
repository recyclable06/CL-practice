import torchvision, torch
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn


num_epochs, batch_size, lr = 5, 8, 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = torch.Generator().manual_seed(1)

torchvision.datasets.CIFAR10.url = (
    "https://hf-mirror.com/datasets/MIT-OL-AI-D/"
    "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
)
transform = v2.Compose([v2.ToTensor()])
tot_train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# train_data = Subset(tot_train_data, range(45000))
# eval_data = Subset(tot_train_data, range(45000,50000))
train_data, eval_data = random_split(
    tot_train_data, [45000, 5000], generator=generator
)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
eval_loader = DataLoader(eval_data,batch_size=batch_size, num_workers=0)

# print(f"训练集大小为：{len(train_data)}")
# image, target = train_data[0]
# print(f"示例元素：图像大小为{image.shape}, 类型为{type(image)}, 标识为{target}，类型为{type(target)}")
# batch1 = next(iter(train_loader))
# print(f"batch len:{len(batch1)}")
# images, targets = batch1
# for idx, (image, target) in enumerate(zip(images, targets)):
#     print(f"图像大小为{image.shape}, 类型为{type(image)},数值类型为{image.dtype}, 标识为{target}，类型为{type(target)}")
#     print(f"数值范围为{image.min():.4f}-{image.max():.4f}")

# print(f"整个batch数值范围为{images.min()}-{images.max()}")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Flatten(), nn.Linear(32*16*16, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.net(x)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# parameter0 = next(model.parameters())[0].detach().clone()
# print(f"初始参数1为：{parameter0}")





sample_image, sample_label = tot_train_data[0]
sample_image = sample_image.to(device).unsqueeze(0)

with torch.no_grad():
    sample_output = model(sample_image)
    print(f"挑选图片标签为{sample_label}")
    print(f"初始logits为{sample_output}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # if epoch == 0:
    #     parameter0_epoch0 = next(model.parameters())[0].detach().clone()
    #     print(f"训练后参数1为：{parameter0_epoch0}")
    #     print(f"变化为：{parameter0_epoch0-parameter0}")

    scheduler.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        total_eval_loss = 0
        for images, targets in eval_loader:
            images, targets = images.to(device), targets.to(device)
            eval_output = model(images)
            eval_loss = criterion(eval_output, targets)
            total_eval_loss += eval_loss
            correct += (eval_output.argmax(dim=-1)==targets).sum().item()

    print(f"epoch {epoch+1}/{num_epochs}, loss: {total_loss/len(train_loader):.4f}, eval_loss: {total_eval_loss/len(eval_loader):.4f}, eval_acc: {correct}/{len(eval_data)}")
    torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, f"ckpt_ep{epoch}.pt")

    if epoch == num_epochs-1:
        with torch.no_grad():
            sample_output_aft_train = model(sample_image)
            print(f"训练后logits：{sample_output_aft_train}")


model_test = CNN().to(device)

path = Path(f"ckpt_ep{num_epochs-1}.pt")
checkpoint = torch.load(path, map_location=device)

model_test.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model_test.eval()
with torch.no_grad():
    sample_output_aft_load = model_test(sample_image)
    print(f"加载后logits：{sample_output_aft_load}")
    print(f"最大差异：{(sample_output_aft_train-sample_output_aft_load).abs().max().item()}")