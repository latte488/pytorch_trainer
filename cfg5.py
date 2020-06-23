import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg5'
epoch_size = 200
dataset_loader = datasets.uiuc_camera_action('data/uiuc_t01_camera_action', batch_size=8)
trainer_type = trainer.Classifier
model = models.LstmNet(input_size=3 * 32 * 32, num_classes=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'