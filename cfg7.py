import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg7'
epoch_size = 200
dataset_loader = datasets.ucf101('data/ucf101')
trainer_type = trainer.Classifier
model = models.ConvLstmNet(num_classes=101)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'
