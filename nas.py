import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
import os
import shutil
from time import time
from torchvision.utils import save_image
from classifier import Classifier
import optuna

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, help='script of python.')
    args = parser.parse_args()
    module_name = args.script.replace('/', '.')[:-3]

    logdir = f'data/{module_name}'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.makedirs(logdir)
    else:
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    module = import_module(module_name)

    def objective(trial):
        model = module.model_build(trial)
        optimizer = module.optimizer_build(model)
        classifier = Classifier(module.device, model, module.criterion, optimizer)
        train_loader, test_loader = module.dataset_loader
        for epoch in range(module.epoch_size - 1):
            train_loss, train_accuracy = classifier.train(train_loader)
            print(f'Epoch {epoch + 1} | Loss {train_loss:.3f} | Acc {train_accuracy:.3f}')
        train_loss, train_accuracy = classifier.train(train_loader)
        writer.add_scalar('Loss/Train', train_loss, trial.number + 1)
        writer.add_scalar('Accuracy/Train', train_accuracy, trial.number + 1)
        test_loss, test_accuracy = classifier.test(test_loader)
        writer.add_scalar('Loss/Test', test_loss, trial.number + 1)
        writer.add_scalar('Accuracy/Test', test_accuracy, trial.number + 1)

        distance = (train_accuracy - test_accuracy) + (train_loss - test_loss)
        distance = max([0, distance])
        return 100.0 - test_accuracy + test_loss + distance
   
    print('Start study')
    study = optuna.create_study()
    study.optimize(objective, n_trials=module.trial_size)

    writer.close()

    print(study.best_params)
    joblib.dump(study, f'{logdir}/study.pkl')

