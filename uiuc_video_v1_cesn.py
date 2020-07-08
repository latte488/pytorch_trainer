import torch
import datasets
import models
from torch.nn.init import calculate_gain, _calculate_correct_fan
import math

trial_size = 100
dataset_loader = datasets.uiuc_video(1, batch_size=8)
epoch_size = 50
device = 'cuda:0'
criterion = torch.nn.CrossEntropyLoss()


def model_build(trial):
    model = models.CESN(8)
    w_res = model.rnn.cell.w_res.data
    size_res = model.rnn.cell.size_res
    fan = _calculate_correct_fan(w_res, 'fan_in')
    gain = calculate_gain('tanh', math.sqrt(5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    for i in range(size_res):
        for j in range(size_res):
            suggest = trial.suggest_uniform(f'w_res[{i}][{j}]', -bound, bound)
            model.rnn.cell.w_res.data[i][j] = suggest
    return model

def optimizer_build(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8, weight_decay=1e-2)

