import torchvision, torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import v2
def build_generator(seed, num):
    return (torch.Generator().manual_seed(seed) for _ in range(num))


def build_CIFAR10_data():

    torchvision.datasets.CIFAR10.url = (
        "https://hf-mirror.com/datasets/MIT-OL-AI-D/"
        "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
    )
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

def build_task_data(tot_train_data, num_task, class_list, train_val_generator, train_loader_generator, batch_size):
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
    return {
        "task_train_data": task_train_data,
        "task_val_data": task_val_data,
        "task_train_loader": task_train_loader,
        "task_val_loader": task_val_loader
    }

