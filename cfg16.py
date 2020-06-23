import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg16'
epoch_size = 200
dataset_loader = datasets.uiuc_camera_action('data/uiuc_t01_camera_action', batch_size=8)
trainer_type = trainer.Classifier
model = models.ConvGRUNet(num_classes=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'
