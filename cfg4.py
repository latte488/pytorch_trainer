import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cifar10_resnet3d18'
epoch_size = 200
dataset_loader = datasets.cifar3d10(root, batch_size=256)
trainer_type = trainer.Classifier
model = models.resnet3d18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'