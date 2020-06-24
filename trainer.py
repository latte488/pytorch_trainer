import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
import os
import shutil
from time import time
from torchvision.utils import save_image

class Trainer:

    def __init__(self, cfg):
        self.model = cfg.model.to(cfg.device)
        self.criterion = cfg.criterion
        self.optimizer = cfg.optimizer
        self.device = cfg.device

        print(f'Model = {self.model}')
        print(f'Criterion = {self.criterion}')
        print(f'Optimizer = {self.optimizer}')
        print(f'Device = {self.device}')

    def accuracy(self, outputs, labels):
        raise NotImplementedError

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
            total_accuracy += self.accuracy(outputs, labels)
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


class Classifier(Trainer):

    def __init__(self, cfg):
        super(Classifier, self).__init__(cfg)

    def accuracy(self, outputs, labels):   
        return outputs.max(1)[1].eq(labels).sum().item() / labels.size(0)

class ImageRegression(Trainer):

    def __init__(self, cfg):
        super(ImageRegression, self).__init__(cfg)
        self.count = 0
        self.imnum = 0
        self.root = cfg.root
    
    def accuracy(self, outputs, labels):
        self.count += 1
        if self.count % 10000 == 0:
            self.imnum += 1
            save_image(outputs.data, f'{self.root}/{self.imnum}_output.png', nrow=4, normalize=True)
            save_image(labels.data, f'{self.root}/{self.imnum}_label.png', nrow=4, normalize=True)
            print(f'{self.imnum} image save.')

        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='module name')
    args = parser.parse_args()
    cfg = import_module(args.cfg.replace('/', '.')[:-3])
    trainer = cfg.trainer_type(cfg)
    root = f'{cfg.root}/runs'
    if os.path.exists(root):
        shutil.rmtree(root)
        os.makedirs(root)
    else:
        os.makedirs(root)
    writer = SummaryWriter(root)
    for epoch in range(cfg.epoch_size):
        print(f'\nEpoch {epoch + 1}')
        train_loader, test_loader = cfg.dataset_loader
        train_loss, train_accuracy = trainer.train(train_loader)
        print(f'Train | Loss {train_loss:.3f} | Accuracy {train_accuracy:.3f}')
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch + 1)
        test_loss, test_accuracy = trainer.test(test_loader)
        print(f'Test  | Loss {test_loss:.3f} | Accuracy {test_accuracy:.3f}')
        writer.add_scalar('Loss/Test', test_loss, epoch + 1)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch + 1)
    writer.close()

