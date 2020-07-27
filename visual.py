import torch
import matplotlib.pyplot as plt

def imsave(save_dir, model, video):
    feature_maps = model.cnn(video)
    for time, images in enumerate(feature_maps):
        for channel, image in enumerate(images):
            plt.imsave(f'{save_dir}/{time}_{channel}.jpg', image.numpy())

def build(model_dir):
    
