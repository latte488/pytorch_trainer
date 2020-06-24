import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg18'
epoch_size = 200
dataset_loader = datasets.cifar10_for_o2o('data/cifar10_for_o2o', batch_size=8)
trainer_type = trainer.ImageRegression
model = models.OriginToOrigin(code_size=128, channels=3, image_size=32)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
device = 'cuda:0'
