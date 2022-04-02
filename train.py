import model
import data
from tqdm import tqdm
import torch
import torch.nn as nn
#import matplotlib
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import numpy.random as npr
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
#import pylab

class TrainConfig:
    # Model hyperparameters
    epochs = 30
    dims = 64
    n_layers = [4,2,2]
    directions = [2]
    learn_rate = 1e-5
    train_pct = 0.8
    valid_pct = 0.1

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)



def gaussian_pdf(x, mu, sigmasq):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * ((x-mu)**2))


def loss_fn(mu, sigmasq, pi, target, num_mixtures=10):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    likelihood_z_x = gaussian_pdf(target.unsqueeze(-1), mu, sigmasq) + 1e-5 # add small positive constant
    prior_z = pi+1e-5  # add small positive constant (to avoid nans)
    losses = (prior_z * likelihood_z_x).sum(axis=-1)
    loss = torch.mean(-torch.log(losses))
    return loss


def train(network, tr_data, va_data, config):
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learn_rate)
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    last_save = -np.inf
    for epoch in range(config.epochs):
        for x,y,cond in tqdm(tr_data):
            optimizer.zero_grad()
            noise = torch.normal(mean=torch.zeros(x.shape[0],1,1))
            pred_params = network(x, cond, noise)
            batch_loss = loss_fn(*pred_params, y)
            batch_loss.backward()
            optimizer.step()
            train_loss_list.append(float(batch_loss.item()))
        va_loss = 0
        for x,y,cond in va_data:
            noise = torch.normal(mean=torch.zeros(x.shape[0],1,1))
            #print(x,y,cond,noise)
            pred_params = network(x, cond, noise)
            #print(pred_params[0][0,0])
            batch_loss = loss_fn(*pred_params, y)
            va_loss += float(batch_loss.item())
        valid_loss_list.append(va_loss)
        if va_loss < best_loss and epoch - last_save > 5: # only save max every 5 epochs
            print("Saving network at epoch", epoch)
            network.save(epoch, va_loss)
            last_save = epoch
            best_loss = va_loss
        print(va_loss)

    plt.plot(train_loss_list,label="Train Loss", color="red")
    plt.plot(valid_loss_list,label="Valid Loss", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("Training and Validation Losses")
    plt.legend()
