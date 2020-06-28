import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
import os
import shutil
from time import time
from torchvision.utils import save_image

class Classifier:
    def __init__(self, device, model, criterion, optimizer):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer

    def _do(self, loader, begin_update, end_update):
        total_loss = 0
        total_accuracy = 0
        for batch_i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            begin_update()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            end_update(loss)
            total_loss += loss.item()
            total_accuracy += outputs.max(1)[1].eq(labels).sum().item() / labels.size(0)
            current_loss = total_loss / (batch_i + 1)
            current_accuracy = 100. * total_accuracy / (batch_i + 1)
        loader_size = len(loader)
        return total_loss / loader_size, 100. * total_accuracy / loader_size

    def train(self, loader):
        def begin_update():
            self.optimizer.zero_grad()
        def end_update(loss):
            loss.backward()
            self.optimizer.step()
        self.model.train()
        return self._do(loader, begin_update, end_update)

    def test(self, loader):
        def begin_update():
            pass
        def end_update(_):
            pass
        self.model.eval()
        with torch.no_grad():
            return self._do(loader, begin_update, end_update)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, help='script of python.')
    args = parser.parse_args()
    module_name = args.script.replace('/', '.')[:-3]

    logdir = f'data/{module_name}/runs'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.makedirs(logdir)
    else:
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    module = import_module(module_name)
    classifier = Classifier(module.device, module.model, module.criterion, module.optimizer)

    print(f'Module name     : {module_name}')
    print(f'Log directory   : {logdir}')
    print(f'Dataset         : {module.dataset_loader[1].dataset.__class__.__name__}')
    print(f'Epoch           : {module.epoch_size}')
    print(f'Batch size      : {module.dataset_loader[1].batch_size}')
    print(f'Device          : {module.device}')
    print(f'Model           : {module.model}')
    print(f'Criterion       : {module.criterion}')
    print(f'Optimizer       : {module.optimizer}')

    for epoch in range(1, module.epoch_size + 1):
        train_loader, test_loader = module.dataset_loader
        print(f'\nEpoch {epoch}')

        train_loss, train_accuracy = classifier.train(train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        print(f'Train | Loss {train_loss:.3f} | Accuracy {train_accuracy:.3f}')

        test_loss, test_accuracy = classifier.test(test_loader)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        print(f'Test  | Loss {test_loss:.3f} | Accuracy {test_accuracy:.3f}')

    writer.close()

