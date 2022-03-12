import model
import data
from tqdm import tqdm
import torch
#import torch.nn as nn
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import numpy.random as npr
from torch.autograd import Variable
from torch.utils.data import DataLoader
#import pylab

np.random.seed(36) # to ensure consistency of train-test split

# Model hyperparameters
EPOCHS = 128
DIMS = 256
N_LAYERS = [8,3,2,2]
DIRECTIONS = [2,2,1]
LEARN_RATE = 1e-5
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1

# Data hyperparameters
WIN_SIZE = 10 # num seconds
SR = 22050  # sample rate
STFT_WIN_SIZE = 256
STFT_HOP_SIZE = 512
NUM_MELS = 256

def gaussian_pdf(x, mu, sigmasq):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * ((x-mu)**2))


def loss_fn(pi, sigmasq, mu, target, num_mixtures=10):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    losses = Variable(torch.zeros_like(target))
    for i in range(num_mixtures):
        likelihood_z_x = gaussian_pdf(target, mu[...,i], sigmasq[..., i])
        prior_z = pi[..., i]
        losses[i] = prior_z * likelihood_z_x
    loss = torch.mean(-torch.log(losses))
    return loss


def train(network, tr_data, va_data):
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARN_RATE)
    train_loss_list = []
    valid_loss_list = []
    for epoch in tqdm(range(EPOCHS)):
        for x,y,cond in tr_data:
            print(x.shape)
            optimizer.zero_grad()
            noise = torch.normal(mean=torch.zeros(x.shape[0],1,1))
            pred_params = network(x, cond, noise)
            batch_loss = loss_fn(pred_params, y)
            batch_loss.backward()
            optimizer.step()
            train_loss_list.append(batch_loss)
        va_loss = 0
        for x,y,cond in va_data:
            noise = torch.normal(size=(x.shape[0],1,1))
            pred_params = network(x, cond, noise)
            batch_loss = loss_fn(pred_params, y)
            va_loss += batch_loss
        valid_loss_list.append(va_loss)
        print(va_loss)

    plt.plot(train_loss_list,label="Train Loss", color="red")
    plt.plot(valid_loss_list,label="Valid Loss", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.title("Training and Validation Losses")
    plt.legend()


df = pd.read_csv("out.csv")
df["class"] = data.encode_classes(df["class"])
size = len(df)
all_indices = np.arange(size)
np.random.shuffle(all_indices)
df = np.array(df)

params = (WIN_SIZE, SR, STFT_WIN_SIZE, STFT_HOP_SIZE, NUM_MELS)

tr_dataset = data.MusicDataset(df[all_indices][:int(size*TRAIN_SIZE)], *params)
va_dataset = data.MusicDataset(df[all_indices][int(size*TRAIN_SIZE):int(size*(TRAIN_SIZE+VALID_SIZE))], *params)
te_dataset = data.MusicDataset(df[all_indices][int(size*TRAIN_SIZE+VALID_SIZE):], *params)
#elem = tr_dataset[1]
#print([e.shape for e in elem[:-1]], elem[-1])
#print(tr_dataset.num_mels, tr_dataset.time_steps)
#quit()
tr_load = DataLoader(tr_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)
va_load = DataLoader(va_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)
te_load = DataLoader(te_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)

network = model.MelNet(DIMS, N_LAYERS, 2, tr_dataset.num_mels, tr_dataset.time_steps, DIRECTIONS)
train(network, tr_load, va_load)
