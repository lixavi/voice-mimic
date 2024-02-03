import torch
from torch.utils.data import Dataset
import numpy as np

class VoiceDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx:idx + self.sequence_length]),
            torch.FloatTensor(self.data[idx + 1:idx + self.sequence_length + 1])
        )

def prepare_dataset(data, sequence_length, batch_size, shuffle=True):
    dataset = VoiceDataset(data, sequence_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
