import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataloader(data_tensor, labels_tensor, batch_size=32):
    dataset = CustomDataset(data_tensor, labels_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
