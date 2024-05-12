import os
import random
import torch
import cv2
import numpy as np
from torchvision.io import read_video
import torchvision.transforms as transforms

## Train / Validation / Test Splitter
def split_dataset(dataset, split_ratio=[0.8, 0.1, 0.1]):
    num_data = len(dataset)
    train_size = int(split_ratio[0] * num_data)
    val_size = int(split_ratio[1] * num_data)
    test_size = num_data - train_size - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

class ShopliftingPreprocessing(object):
    def __init__(self, output_size=(120, 120)):
        self.output_size = output_size
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, video):
        resized_video = np.zeros((3, self.output_size[0], self.output_size[1], 30))
        
        # I choose the get first 16 frames, but ofc there are several other techniques...
        for i in range(30):            
            
            frame_np = video[i].numpy()

            frame = cv2.resize(frame_np, self.output_size)
            frame = np.reshape(frame, (3, self.output_size[0], self.output_size[1]))
            frame = frame / 255
            frame = frame.reshape(3, self.output_size[0], self.output_size[1])
            
            resized_video[:, :, :, i] = frame

        return resized_video

## Dataloader Class
class ShopliftingDataLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        self.transform = transform

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for sample_name in sorted(os.listdir(class_dir)):
                sample_path = os.path.join(class_dir, sample_name)
                samples.append((sample_path, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, target = self.samples[idx]
        video, audio, info = read_video(sample_path)
        if self.transform:
            video = self.transform(video)
        return video, target