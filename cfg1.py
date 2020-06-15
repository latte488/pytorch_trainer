import torch
import trainer
import dataset
import model

root = 'data/mnist_lenet'
epoch_size = 10
dataset_loader = dataset.mnist(root, batch_size=64)
trainer_type = trainer.Classifier
model = model.LeNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
device = 'cuda:0'
