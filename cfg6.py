import torch
import torchvision
import trainer
import datasets
import models

root = 'data/cfg6'
epoch_size = 10
dataset_loader = datasets.action_mnist('data/action_mnist', batch_size=8)
trainer_type = trainer.Classifier
model = models.LstmNet(input_size=3 * 64 * 64, num_classes=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8, weight_decay=1e-2)
device = 'cuda:0'
