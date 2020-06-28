import cv2
import torch
import torchvision
import toml
import numpy as np

class UiucVideo(torchvision.datasets.VisionDataset):
    def __init__(self, root, number, transforms=None, transform=None, target_transform=None):
        super(UiucVideo, self).__init__(root, transforms, transform, target_transform)
        self.root = f'{self.root}/uiuc_T{number:02}_camera_action_dataset'
        self.number = number
        label_path = f'{self.root}/0.toml'
        label_dist = toml.load(label_path)
        self.action_label_dist = label_dist['Action label']
        self.path_and_label = list(label_dist['Video label'].items())
        self.size = len(self.path_and_label)

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

class UiucVideoV1(UiucVideo):
    def __init__(self, root, train=True, **kwargs):
        super(UiucVideoV1, self).__init__(root, 1, **kwargs)
        if train:
            self.number = 12800
            self.offset = 0
        else:
            self.number = 12800 // 5
            self.offset = 12800 * 4 // 5

    def __getitem__(self, index):
        return super(UiucVideoV1, self).__getitem__(self.offset + index)

    def __len__(self):
        return self.number

class UiucVideoV2(UiucVideo):
    def __init__(self, root, train=True, **kwargs):
        super(UiucVideoV2, self).__init__(root, 1 if train else 2, **kwargs)

class UiucVideoV3(torchvision.datasets.VisionDataset):
    def __init__(self, root, train=True, **kwargs):
        super(UiucVideoV3, self).__init__(root, **kwargs)
        self.datasets = [UiucVideo(root, i + 1, **kwargs) for i in range(25)]
        if train:
            self.num_per_type = 256 * 4
            self.offset = 0
        else:
            self.num_per_type = 64 * 4
            self.offset = 256 * 2

    def __getitem__(self, index):
        uiuc_type = index // self.num_per_type
        uiuc_index = self.offset + index % self.num_per_type
        return self.datasets[uiuc_type][uiuc_index]

    def __len__(self):
        return self.num_per_type * 25
         
def uiuc_video(version, batch_size, root='data/uiuc'):

    dataset_classes = [
        UiucVideoV1,
        UiucVideoV2,
        UiucVideoV3,
    ]

    dataset_class = dataset_classes[version - 1]

    transform = torchvision.transforms.ToTensor()
    train_dataset = dataset_class(
        root=root, 
        train=True,
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )   
    test_dataset = dataset_class(
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
    
