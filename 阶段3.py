import torchvision, torch
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset, Dataset
from torch import nn


num_epochs, batch_size, lr = 5, 8, 0.001
torch.manual_seed(1)
cache_size=200
logits_alpha = 0.5
cache_beta = 0.5

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

class Data_Logits_Set(Dataset):
    def __init__(self, data, logits):
        super().__init__()
        self.data = data
        self.logits = logits
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.logits[index]



#阶段2，task预处理
num_task=3
class_list = [[0,1],[2,3],[4,5]]

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

preserve_cache = []

# cur_cache_ratio = [1,1]
# cur_cache_num = []
# sum_cur_cache = sum(cur_cache_ratio)
# for i in range(task-1):
#     cur_cache_num.append(int(batch_size * cur_cache_ratio[i] / sum_cur_cache))
# cur_cache_num.append(batch_size-sum(cur_cache_num))




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
logits_loss_fn = nn.MSELoss(reduction='mean')


def train(model, criterion, optimizer, train_loader, device, cache_loader=None, logits_alpha=0.5, cache_beta=0.5,logits_loss_fn=nn.MSELoss(reduction='mean')):
    model.train()
    total_train_loss = 0
    cur_task_loss ,total_cache_loss, total_logits_loss = 0, 0, 0
    len_train_sample, len_cur_sample, len_cache_sample = 0, 0, 0
    if cache_loader: reply_cache_iter = iter(cache_loader)
    for images, targets in train_loader:

        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        cur_task_loss += loss.item() * images.shape[0]
        num_sample = targets.shape[0]
        len_cur_sample += targets.shape[0]

        if cache_loader:
            try:
                (cache_image, cache_target), cache_logits = next((reply_cache_iter))
            except StopIteration:
                reply_cache_iter = iter(cache_loader)
                (cache_image, cache_target), cache_logits = next((reply_cache_iter))

            cache_image, cache_target = cache_image.to(device), cache_target.to(device)
            cache_logits = cache_logits.to(device)
            
            cache_output = model(cache_image)
            cache_loss = criterion(cache_output, cache_target)
            logits_loss = logits_loss_fn(cache_output, cache_logits)
            loss += logits_loss * logits_alpha + cache_loss * cache_beta
            total_cache_loss += cache_loss.item() * cache_image.shape[0]
            total_logits_loss += logits_loss.item() * cache_logits.shape[0]

            num_sample += cache_image.shape[0]
            len_cache_sample += cache_image.shape[0]

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * num_sample
        len_train_sample += num_sample


    return {
                "mean_train_loss": total_train_loss / len_train_sample,
                "len_train_sample": len_train_sample,
                "mean_cur_loss": cur_task_loss / len_cur_sample,
                "mean_cache_loss": total_cache_loss / len_cache_sample if cache_loader else 0,
                "mean_logits_loss": total_logits_loss / len_cache_sample if cache_loader else 0
            }
            


@torch.no_grad()    
def evaluate(model,  criterion, val_loader, device, if_cache_loader=False):
    model.eval()
    correct = 0
    bin_count = torch.zeros(10, dtype=torch.int64, device=device)
    total_eval_loss = 0
    
    for batch in val_loader:
        if if_cache_loader == False:
            images, targets = batch
        else:
            (images, targets), _ =batch
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        predictions = output.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        bin_count += torch.bincount(predictions, minlength=10)
        total_eval_loss += criterion(output, targets) * targets.shape[0]

    return correct, total_eval_loss, bin_count


def update_cache(train_data, preserve_cache, cache_size, cache_sample_generator, model, device):

    num_task = len(preserve_cache)+1
    per_cache_num = int(cache_size/num_task)
    last_cache_num = cache_size - (num_task - 1) * per_cache_num
    new_sample_data, _ = random_split(
        train_data,
        [last_cache_num, len(train_data)-last_cache_num],
        generator=cache_sample_generator
    )

    new_sample_images = torch.stack([new_sample_data[i][0] for i in range(len(new_sample_data))]).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(new_sample_images)
    new_set = Data_Logits_Set(new_sample_data, logits)
    preserve_cache.append(new_set)

    for i in range(num_task-1):
        old_cache = preserve_cache[i]
        preserve_cache[i], _ =random_split(
            old_cache,
            [per_cache_num, len(old_cache) - per_cache_num],
            generator=cache_sample_generator
        )


for task in range(num_task):
    train_loader = task_train_loader[task]
    val_loader = task_val_loader[task]
    val_data = task_val_data[task]
    train_data = task_train_data[task]

    cache_loader = None
    if task>0:
        cache_loader = DataLoader(ConcatDataset(preserve_cache), batch_size=batch_size, shuffle=True, num_workers=0, generator=cache_loader_generator)


    # mix_data = ConcatDataset(preserve_cache_image + [train_data])
    # train_loader = DataLoader(mix_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    print(f"\n当前进入task{task}, 学习类别{class_list[task]}")
    for epoch in range(num_epochs):

        
        result = train(model, criterion, optimizer, train_loader, device, cache_loader, logits_alpha, cache_beta, logits_loss_fn)

        # scheduler.step()

        correct, total_eval_loss, _ = evaluate(model, criterion, val_loader, device)

        print(f"epoch {epoch+1}/{num_epochs}, 总loss: {result["mean_train_loss"]:.4f}, 当前task图像训练loss: {result["mean_cur_loss"]:.4f}, cache的图像loss: {result["mean_cache_loss"]:.4f}, logits的loss： {result["mean_logits_loss"]:.4f},\neval_loss: {total_eval_loss/len(val_data):.4f}, eval_acc: {correct}/{len(val_data)}")
        # torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, f"ckpt_ep{epoch}.pt")
        
        
        if task>0:
            val_correct, _, _ = evaluate(model, criterion, task_val_loader[task-1], device)
            cache_correct, _, _ = evaluate(model, criterion, cache_loader, device, if_cache_loader=True)

            print(f"对已有task的cache_acc：{cache_correct}/{cache_size}   ,对上一个task的val_acc： {val_correct}/{len(task_val_data[task-1])}")



    for pre_task in range(task+1):
        val_loader = task_val_loader[pre_task]
        val_data = task_val_data[pre_task]

        correct, _, bin_count =  evaluate(model, criterion, val_loader, device)

        CL_task_acc[task][pre_task]=correct/len(val_data)
        print(f"第{task}轮task学习结束，对第{pre_task}轮task的acc：{correct}/{len(val_data)} = {correct/len(val_data)}, 预测分布为{bin_count.tolist()}")




    update_cache(train_data, preserve_cache, cache_size, cache_sample_generator, model, device)

    torch.save({"task":task, 
                "model_state_dict":model.state_dict(), 
                "optimizer_state_dict":optimizer.state_dict(), 
                "train_val_generator_state":train_val_generator.get_state(),
                "train_loader_generator_state":train_loader_generator.get_state(),
                "cache_loader_generator_state":cache_loader_generator.get_state(),
                "cache_sample_generator_state":cache_sample_generator.get_state()
                }, f"ckpt_task{task}.pt")


print(f"CL_task_acc:{CL_task_acc}")





task2_val_loader = task_val_loader[1]
model.eval()
in_task2_correct = 0
task2_mask = torch.zeros(10, dtype=torch.int, device=device)
task2_mask[class_list[1]] = 1
with torch.no_grad():
    for images, targets in task2_val_loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        output = output.masked_fill(task2_mask==0, float("-inf"))
        in_task2_correct += (output.argmax(dim=-1) == targets).sum().item()

print(f"仅在task2的类别内计算模型对task2的判断能力，acc: {in_task2_correct}/{len(task_val_data[1])}")