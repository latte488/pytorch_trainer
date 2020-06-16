import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from tqdm import tqdm

class Trainer:

    def __init__(self, cfg):
        self.model = cfg.model.to(cfg.device)
        self.criterion = cfg.criterion
        self.optimizer = cfg.optimizer
        self.device = cfg.device

    def accuracy(self, outputs, labels):
        raise NotImplementedError

    def _do(self, loader, begin_update, end_update, desc):
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        progress = tqdm(total=len(loader), desc=desc)
        for batch_i, (inputs, labels) in enumerate(loader):
            begin_update()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            end_update(loss)
            total_loss += loss.item()
            total_accuracy += self.accuracy(outputs, labels)
            total_number += labels.size(0)
            current_loss = total_loss / len(loader)
            current_accuracy = 100. * total_accuracy / total_number
            progress.set_description(f'{desc} | Loss {current_loss:.3f} | Acc {current_accuracy:.3f}')
            progress.update(1)
        progress.close()
      
        return total_loss / len(loader), 100. * total_accuracy / total_number

    def train(self, loader):
        def begin_update():
            self.optimizer.zero_grad()
        def end_update(loss):
            loss.backward()
            self.optimizer.step()
        self.model.train()
        return self._do(loader, begin_update, end_update, 'Train')

    def test(self, loader):
        def begin_update():
            pass
        def end_update(_):
            pass
        self.model.eval()
        with torch.no_grad():
            return self._do(loader, begin_update, end_update, 'Test ')

class Classifier(Trainer):

    def __init__(self, cfg):
        super(Classifier, self).__init__(cfg)

    def accuracy(self, outputs, labels):   
        return outputs.max(1)[1].eq(labels).sum().item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='module name')
    args = parser.parse_args()
    cfg = import_module(args.cfg.replace('/', '.')[:-3])
    trainer = cfg.trainer_type(cfg)
    writer = SummaryWriter(f'{cfg.root}/runs')
    for epoch in range(cfg.epoch_size):
        print(f'Epoch {epoch}:')
        train_loader, test_loader = cfg.dataset_loader
        train_loss, train_accuracy = trainer.train(train_loader)
        writer.add_scalar('Train-Loss', train_loss, epoch + 1)
        writer.add_scalar('Train-Accuracy', train_accuracy, epoch + 1)
        test_loss, test_accuracy = trainer.test(test_loader)
        writer.add_scalar('Test-Loss', test_loss, epoch + 1)
        writer.add_scalar('Test-Accuracy', test_loss, epoch + 1)
    writer.close()

