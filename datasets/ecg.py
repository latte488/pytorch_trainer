import cv2
import torch
import torchvision
import numpy as np

root = 'data/ecg'
mitbih_train = f'{root}/mitbih_train.csv'
mitbih_test = f'{root}/mitbih_test.csv'
ptbdb_normal = f'{root}/ptbdb_normal.csv'
ptbdb_abnormal = f'{root}/ptbdb_abnormal.csv'

class ECG(torchvision.datasets.VisionDataset):

    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None):
        super(ECG, self).__init__(transforms, transform, target_transform)
        path = mitbih_train if train else mitbih_test
        self.data = np.loadtxt(path, delimiter=',')
        
    def __getitem__(self, index):
        return self.data[index][:-1].astype(np.float32), int(self.data[index][-1])

    def __len__(self):
        return self.data.shape[0]

def ecg(root, batch_size):
    transform = torchvision.transforms.ToTensor()
    train_dataset = ECG(
        root=root, 
        train=True,
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )   
    test_dataset = ECG(
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

