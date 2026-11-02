import torch
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset

def build_cache_loader(preserve_cache, batch_size, cache_loader_generator):
    return DataLoader(ConcatDataset(preserve_cache), batch_size=batch_size, shuffle=True, num_workers=0, generator=cache_loader_generator)



class Data_Logits_Set(Dataset):
    def __init__(self, data, logits):
        super().__init__()
        self.data = data
        self.logits = logits
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.logits[index]

def update_cache(train_data, preserve_cache, cache_size, cache_sample_generator, model, device, method="derpp"):

    num_task = len(preserve_cache)+1
    per_cache_num = int(cache_size/num_task)
    last_cache_num = cache_size - (num_task - 1) * per_cache_num
    new_sample_data, _ = random_split(
        train_data,
        [last_cache_num, len(train_data)-last_cache_num],
        generator=cache_sample_generator
    )

    if method == "derpp":
        new_sample_images = torch.stack([new_sample_data[i][0] for i in range(len(new_sample_data))]).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(new_sample_images)
        new_set = Data_Logits_Set(new_sample_data, logits)
        preserve_cache.append(new_set)
    else:
        preserve_cache.append(new_sample_data)

    for i in range(num_task-1):
        old_cache = preserve_cache[i]
        preserve_cache[i], _ =random_split(
            old_cache,
            [per_cache_num, len(old_cache) - per_cache_num],
            generator=cache_sample_generator
        )
