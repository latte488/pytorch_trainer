import torch
import datasets
import models

dataset_loader = datasets.uiuc_video(3, batch_size=8)
epoch_size = 200
device = 'cuda:0'
model = models.ConvTanhRNNNet(num_classes=8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.8, weight_decay=1e-2)
