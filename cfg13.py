import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg13'
epoch_size = 10
dataset_loader = datasets.ecg('', batch_size=64)
trainer_type = trainer.Classifier
model = models.DNN(187, 5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'
