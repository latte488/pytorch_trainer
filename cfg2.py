import torch
import torchvision
import trainer
import dataset

root = 'data/cifar10_resnet18'
epoch_size = 10
dataset_loader = dataset.cifar10(root, batch_size=64)
trainer_type = trainer.Classifier
model = torchvision.models.resnet18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
device = 'cuda:0'
