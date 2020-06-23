import torch
import torchvision
import trainer
import datasets

root = 'data/cifar10_resnet18'
epoch_size = 50
dataset_loader = datasets.cifar10(root, batch_size=256)
trainer_type = trainer.Classifier
model = torchvision.models.resnet18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
device = 'cuda:0'
