import torch
import trainer
import datasets
import models

root = 'data/fashion_mnist_lenet'
epoch_size = 10
dataset_loader = datasets.fashion_mnist(root, batch_size=64)
trainer_type = trainer.Classifier
model = models.LeNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
device = 'cuda:0'
