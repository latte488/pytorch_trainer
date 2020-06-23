import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg14'
epoch_size = 200
dataset_loader = datasets.action_mnist('data/action_mnist', batch_size=8)
trainer_type = trainer.Classifier
model = models.GRUNet(input_size=3 * 64 * 64, output_size=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'
