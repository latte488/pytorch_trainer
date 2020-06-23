import cv2
import torch
import torchvision
import toml
import numpy as np

class ActionMNIST(torchvision.datasets.VisionDataset):

    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None):
        super(ActionMNIST, self).__init__(root, transforms, transform, target_transform)
        self.root = f'{root}/train' if train else f'{root}/test'
        label_path = f'{self.root}/0.toml'
        label_dist = toml.load(label_path)
        self.action_label_dist = label_dist['Action label']
        self.path_and_label = list(label_dist['Video label'].items())
        self.size = 8 * 1000 if train else 8 * 250

    def __getitem__(self, index):
        video_name, label = self.path_and_label[index]
        video_path = f'{self.root}/{video_name}'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('No file in path. Check the `root` path.')
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video = []
        for _ in range(int(frame_number)):
            ok, image = cap.read()
            if not ok: 
                raise ValueError('Failed read video.')
            h, w, c = 0, 1, 2
            video.append(image.transpose(c, h, w))
        return np.array(video).astype(np.float32), self.action_label_dist[label]

    def __len__(self):
        return self.size

def action_mnist(root, batch_size):
    transform = torchvision.transforms.ToTensor()
    train_dataset = ActionMNIST(
        root=root, 
        train=True,
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )   
    test_dataset = ActionMNIST(
        root=root,
        train=False,
        transform=transform
    )   
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )   
    return train_loader, test_loader

