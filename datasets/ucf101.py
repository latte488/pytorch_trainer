import cv2
import torch
import torchvision
import numpy as np

class UCF101(torchvision.datasets.VisionDataset):

    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None):
        super(UCF101, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.list_dir = f'{self.root}/ucfTrainTestlist'
        self.map = {}
        with open(f'{self.list_dir}/classInd.txt') as f:
            for line in f:
                key, value = tuple(line.strip().split(' '))
                self.map[value] = int(key) - 1
        dist = {}
        name = 'trainlist' if train else 'testlist'
        for i in range(1, 4):
            path = f'{self.list_dir}/{name}{i:02d}.txt'
            with open(path) as f:
                for line in f:
                    label_path  = line.strip().split(' ')[0]
                    label, path = tuple(label_path.split('/'))
                    dist[path] = self.map[label]
        self.data = list(dist.items())
        self.size = len(self.data)

    def __getitem__(self, index):
        video_name, label = self.data[index]
        video_path = f'{self.root}/{video_name}'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('No file in path. Check the `root` path.')
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video = []
        for _ in range(int(frame_number)):
            ok, image = cap.read()
            if not ok:
                break
            h, w, c = 0, 1, 2
            image = cv2.resize(image, (80, 60))
            video.append(image.transpose(c, h, w))
        return np.array(video).astype(np.float32), int(label)

    def __len__(self):
        return self.size

def ucf101(root):
    transform = torchvision.transforms.ToTensor()
    train_dataset = UCF101(
        root=root, 
        train=True,
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True
    )   
    test_dataset = UCF101(
        root=root,
        train=False,
        transform=transform
    )   
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )   
    return train_loader, test_loader

ucf101_frame_size = (320 // 4, 240 // 4) # (w, h)
