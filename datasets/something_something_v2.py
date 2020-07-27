import cv2
import torch
import torchvision
import json
import numpy as np
import os
import shutil
import joblib
import matplotlib.pyplot as plt
import statistics
import math

ROOT = 'data/something-something-v2'
VIDEO_DIR = f'{ROOT}/20bn-something-something-v2'
TRAIN_PATH = f'{ROOT}/something-something-v2-train.json'
TEST_PATH = f'{ROOT}/something-something-v2-test.json'
VALIDATION_PATH = f'{ROOT}/something-something-v2-validation.json'
LABELS_PATH = f'{ROOT}/something-something-v2-labels.json'


def video_to_numpy(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('No file in path. Check the `root` path.')
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video = []
    for _ in range(int(frame_number)):
        ok, image = cap.read()
        if not ok:
            break
        h, w = image.shape[:2]
        image = cv2.resize(image, (h // 6, w // 6), interpolation=cv2.INTER_AREA)
        h, w, c = 0, 1, 2
        video.append(image.transpose(c, h, w))
    return np.array(video).astype(np.float32)

def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def train_to_npy():
    mkdir(f'{ROOT}/train')
    with open(TRAIN_PATH) as fp:
        trains = json.load(fp)
    with open(LABELS_PATH) as fp:
        labels = json.load(fp)
    for train in trains:
        number = train['id']
        video_path = f'{VIDEO_DIR}/{number}.webm'
        video = video_to_numpy(video_path)
        template = train['template']
        template = template.replace('[', '')
        template = template.replace(']', '')
        label = labels[template]
        np.save(f'{ROOT}/train/{number}_{label}', video)

def validation_to_npy():
    mkdir(f'{ROOT}/validation')
    with open(VALIDATION_PATH) as fp:
        validations = json.load(fp)
    with open(LABELS_PATH) as fp:
        labels = json.load(fp)
    for validation in validations:
        number = validation['id']
        video_path = f'{VIDEO_DIR}/{number}.webm'
        video = video_to_numpy(video_path)
        template = validation['template']
        template = template.replace('[', '')
        template = template.replace(']', '')
        label = labels[template]
        np.save(f'{ROOT}/train/{number}_{label}', video)
    
def test_to_npy():
    mkdir(f'{ROOT}/test')
    with open(TEST_PATH) as fp:
        tests = json.load(fp)
    for test in tests:
        number = test['id']
        video_path = f'{VIDEO_DIR}/{number}.webm'
        video = video_to_numpy(video_path)
        np.save(f'{ROOT}/test/{number}', video)

class SomethingSomethingV2(torchvision.datasets.VisionDataset):

    def __init__(self, root, data_type='train', transforms=None, transform=None, target_transform=None):
        super(SomethingSomethingV2, self).__init__(root, transforms, transform, target_transform)

        self.video_dir = f'{root}/20bn-something-something-v2'
        self.train_path = f'{root}/something-something-v2-train.json'
        self.test_path = f'{root}/something-something-v2-test.json'
        self.validation_path = f'{root}/something-something-v2-validation.json'
        self.labels_path = f'{root}/something-something-v2-labels.json'

        with open(self.train_path) as fp:
            self.train = json.load(fp)
        with open(self.test_path) as fp:
            self.test = json.load(fp)
        with open(self.validation_path) as fp:
            self.validation = json.load(fp)
        with open(self.labels_path) as fp:
            self.labels = json.load(fp)

        if data_type == 'train':
            self.getitem = self._getitem_train
            self.size = len(list(self.train))
        elif data_type == 'test':
            self.getitem = self._getitem_test
            self.size = len(list(self.test))
        elif data_type == 'validation':
            self.getitem = self._getitem_validation
            self.size = len(list(self.validation))
        else:
            raise ValueError('data_type is one of `train`, `test`, `validation`.')

    def _getitem_train(self, index):
        number = self.train[index]['id']
        video_path = f'{self.video_dir}/{number}.webm'
        video = video_to_numpy(video_path)
        template = self.train[index]['template'].replace('[', '').replace(']', '')
        label = int(self.labels[template])
        return video, label

    def _getitem_test(self, index):
        number = self.test[index]['id']
        video_path = f'{self.video_dir}/{number}.webm'
        video = video_to_numpy(video_path)
        return video

    def _getitem_validation(self, index):
        number = self.validation[index]['id']
        video_path = f'{self.video_dir}/{number}.webm'
        video = video_to_numpy(video_path)
        template = self.validation[index]['template'].replace('[', '').replace(']', '')
        label = int(self.labels[template])
        return video, label

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return self.size

    def train_id(self, index):
        return self.train[index]['id']

def something_something_v2():
    batch_size = 1
    root = 'data/something-something-v2'
    transform = torchvision.transforms.ToTensor()
    train_dataset = SomethingSomethingV2(
        root=root, 
        data_type='train',
        transform=transform
    )   
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )   
    test_dataset = SomethingSomethingV2(
        root=root,
        data_type='validation',
        transform=transform
    )   
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )   
    return train_loader, test_loader

"""
dataset = SomethingSomethingV2(
    root='data/something-something-v2',
    data_type='train',
    transform=torchvision.transforms.ToTensor()
)

data = []
for number, (video, label) in enumerate(dataset):
    t, c, h, w = video.shape
    data.append((t, h, w, number, label))
joblib.dump(data, 'something_v2.txt')
data = joblib.load('something_v2.txt')
ws = []
hs = []
for d in data:
    
    ws.append(d[2])
    hs.append(d[1])

print(f'Min     {min(ws)}')
print(f'Max     {max(ws)}')
print(f'Mean    {statistics.mean(ws)}')
print(f'Median  {statistics.median(ws)}')
print(f'Mode    {statistics.mode(ws)}')
print(f'stdev   {statistics.stdev(ws)}')
frame_number_label = joblib.load('frame_number_label.txt')
sorted_fnl = sorted(frame_number_label, key=lambda x:x[0])
for i in range(5):
    min_frame, min_number, min_label = sorted_fnl[i]
    print(f'Min {i} : frame {min_frame} | number {min_number} | label {min_label}')

for i in range(5):
    max_frame, max_number, max_label = sorted_fnl[-1 - i]
    print(f'Max {i} : frame {max_frame} | number {max_number} | label {max_label}')

frames = [frame for frame, _, _ in frame_number_label]
print(f'Mean    {statistics.mean(frames)}')
print(f'Median  {statistics.median(frames)}')
print(f'Mode    {statistics.mode(frames)}')
print(f'stdev   {statistics.stdev(frames)}')
"""
