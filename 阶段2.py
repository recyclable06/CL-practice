import torchvision, torch
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torch import nn


num_epochs, batch_size, lr = 5, 8, 0.001
torch.manual_seed(1)
cache_size=200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_val_generator = torch.Generator().manual_seed(1)
train_loader_generator = torch.Generator().manual_seed(1)
cache_loader_generator = torch.Generator().manual_seed(1)
cache_sample_generator = torch.Generator().manual_seed(1)

torchvision.datasets.CIFAR10.url = (
    "https://hf-mirror.com/datasets/MIT-OL-AI-D/"
    "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
)
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
tot_train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# train_data = Subset(tot_train_data, range(45000))
# eval_data = Subset(tot_train_data, range(45000,50000))
# train_data, eval_data = random_split(
#     tot_train_data, [45000, 5000], generator=generator
# )
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
# val_loader = DataLoader(eval_data,batch_size=batch_size, num_workers=0)

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





#阶段2，task预处理
num_task=2
class_list = [[0,1],[2,3]]

task_tot_data = [Subset(tot_train_data, [idx for idx, label in enumerate(tot_train_data.targets) if label in class_list[task]]) for task in range(num_task)]

task_train_data, task_val_data, task_train_loader, task_val_loader= [], [], [], []
for task in range(num_task):
    task_train_size = int(len(task_tot_data[task])*0.9)
    task_val_size = len(task_tot_data[task]) - task_train_size

    cur_task_train_data, cur_task_val_data = random_split(
        task_tot_data[task],
        [task_train_size,task_val_size],
        generator=train_val_generator
    )
    task_train_data.append(cur_task_train_data)
    task_val_data.append(cur_task_val_data)
    task_train_loader.append(DataLoader(cur_task_train_data, batch_size=batch_size, shuffle=True, generator=train_loader_generator))
    task_val_loader.append(DataLoader(cur_task_val_data, batch_size=batch_size))

CL_task_acc = [[None for _ in range(num_task)] for _ in range(num_task)]

preserve_cache_image = []

cur_cache_ratio = [1,1]
cur_cache_num = []
sum_cur_cache = sum(cur_cache_ratio)
for i in range(task-1):
    cur_cache_num.append(int(batch_size * cur_cache_ratio[i] / sum_cur_cache))
cur_cache_num.append(batch_size-sum(cur_cache_num))


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





# sample_image, sample_label = tot_train_data[0]
# sample_image = sample_image.to(device).unsqueeze(0)

# with torch.no_grad():
#     sample_output = model(sample_image)
#     print(f"挑选图片标签为{sample_label}")
#     print(f"初始logits为{sample_output}")

for task in range(num_task):
    train_loader = task_train_loader[task]
    val_loader = task_val_loader[task]
    val_data = task_val_data[task]
    train_data = task_train_data[task]

    if task>0:
        cache_loader = DataLoader(preserve_cache_image[task-1], batch_size=batch_size, shuffle=True, num_workers=0, generator=cache_loader_generator)


    # mix_data = ConcatDataset(preserve_cache_image + [train_data])
    # train_loader = DataLoader(mix_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    print(f"\n当前进入task{task}, 学习类别{class_list[task]}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        len_train_sample = 0
        if task>0: reply_cache_iter = iter(cache_loader)

        for images, targets in train_loader:

            if task>0:
                try:
                    cache_image, cache_target = next((reply_cache_iter))
                except StopIteration:
                    reply_cache_iter = iter(cache_loader)
                    cache_image, cache_target = next((reply_cache_iter))

                images = torch.cat([images, cache_image], dim=0)
                targets = torch.cat([targets, cache_target], dim=0)

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.shape[0]
            len_train_sample += targets.shape[0]

        # if epoch == 0:
        #     parameter0_epoch0 = next(model.parameters())[0].detach().clone()
        #     print(f"训练后参数1为：{parameter0_epoch0}")
        #     print(f"变化为：{parameter0_epoch0-parameter0}")

        # scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            total_eval_loss = 0
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                eval_output = model(images)
                eval_loss = criterion(eval_output, targets)
                total_eval_loss += eval_loss
                correct += (eval_output.argmax(dim=-1)==targets).sum().item()

        print(f"epoch {epoch+1}/{num_epochs}, loss: {total_loss/len_train_sample:.4f}, eval_loss: {total_eval_loss/len(val_loader):.4f}, eval_acc: {correct}/{len(val_data)}")
        # torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, f"ckpt_ep{epoch}.pt")
        
        
        if task>0:
            model.eval()
            val_correct = 0
            cache_correct = 0
            with torch.no_grad():
                for images, targets in task_val_loader[task-1]:
                    images, targets = images.to(device), targets.to(device)
                    val_correct += (model(images).argmax(dim=-1)==targets).sum().item()
                for images, targets in cache_loader:
                    images, targets = images.to(device), targets.to(device)
                    cache_correct += (model(images).argmax(dim=-1)==targets).sum().item()
            print(f"对上一个task的cache_acc：{cache_correct}/{cache_size}   ,val_acc： {val_correct}/{len(task_val_data[task-1])}")

        # if epoch == num_epochs-1:
        #     with torch.no_grad():
        #         sample_output_aft_train = model(sample_image)
        #         print(f"训练后logits：{sample_output_aft_train}")

    for pre_task in range(task+1):
        val_loader = task_val_loader[pre_task]
        val_data = task_val_data[pre_task]

        model.eval()
        correct = 0
        bin_count = torch.zeros(10, dtype=torch.int64, device=device)
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images).argmax(dim=-1)
                correct += (predictions == targets).sum().item()
                bin_count += torch.bincount(predictions, minlength=10)

        CL_task_acc[task][pre_task]=correct/len(val_data)
        print(f"第{task}轮task学习结束，对第{pre_task}轮task的acc：{correct}/{len(val_data)} = {correct/len(val_data)}, 预测分布为{bin_count.tolist()}")


    sample_data, _ = random_split(
        train_data,
        [cache_size, len(train_data)-cache_size],
        generator=cache_sample_generator
    )
    preserve_cache_image.append(sample_data)

print(f"CL_task_acc:{CL_task_acc}")
    


# model_test = CNN().to(device)

# path = Path(f"ckpt_ep{num_epochs-1}.pt")
# checkpoint = torch.load(path, map_location=device)

# model_test.load_state_dict(checkpoint["model_state_dict"])
# # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# model_test.eval()
# with torch.no_grad():
#     sample_output_aft_load = model_test(sample_image)
#     print(f"加载后logits：{sample_output_aft_load}")
#     print(f"最大差异：{(sample_output_aft_train-sample_output_aft_load).abs().max().item()}")