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
import warnings

np.random.seed(36) # to ensure consistency of train-test split
torch.set_default_tensor_type("torch.cuda.FloatTensor")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Model hyperparameters
EPOCHS = 30
DIMS = 64
N_LAYERS = [4,2,2]
DIRECTIONS = [2]
LEARN_RATE = 1e-5
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1

# Data hyperparameters
WIN_SIZE = 6 # num seconds
SR = 11025  # sample rate
STFT_WIN_SIZE = 256
STFT_HOP_SIZE = 1024
NUM_MELS = 128

def gaussian_pdf(x, mu, sigmasq):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    #print(x.shape, mu.shape, sigmasq.shape)
    return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * ((x-mu)**2))


def loss_fn(mu, sigmasq, pi, target, num_mixtures=10):
    # adapted from https://mikedusenberry.com/mixture-density-networks
    #losses = Variable(torch.zeros_like(target))
    #print(target.shape, mu.shape, sigmasq.shape, pi.shape)
    #for i in range(num_mixtures):
        #print(mu[...,i].shape)
    likelihood_z_x = gaussian_pdf(target.unsqueeze(-1), mu, sigmasq) + 1e-5 # add small positive constant
    #print(torch.any(torch.isnan(pi)))
    #print(torch.where(torch.isnan(pi)))
    #print(torch.any(torch.isnan(likelihood_z_x)))
    prior_z = pi+1e-5  # add small positive constant (to avoid nans)
    losses = (prior_z * likelihood_z_x).sum(axis=-1)
    loss = torch.mean(-torch.log(losses))
    #print(prior_z.min(), likelihood_z_x.min(), prior_z.mean(), likelihood_z_x.mean())
    #print(torch.isnan(loss), loss)
    return loss


def train(network, tr_data, va_data):
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARN_RATE)
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    last_save = -np.inf
    for epoch in range(EPOCHS):
        for x,y,cond in tqdm(tr_data):
            #print("example shape", x.shape)
            optimizer.zero_grad()
            #with autograd.detect_anomaly():
            noise = torch.normal(mean=torch.zeros(x.shape[0],1,1))
            #print(x,y,cond,noise)
            pred_params = network(x, cond, noise)
            #print(pred_params[0][0,0])
            batch_loss = loss_fn(*pred_params, y)
            batch_loss.backward()
            optimizer.step()
            #############
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
print("calculated size", tr_dataset.num_mels, tr_dataset.time_steps)
#quit()
tr_load = DataLoader(tr_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)
va_load = DataLoader(va_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)
te_load = DataLoader(te_dataset, batch_size=None, batch_sampler=None, shuffle=False, num_workers=0)

network = model.MelNet(DIMS, N_LAYERS, 2, tr_dataset.num_mels, DIRECTIONS)
try:
    network.load(glob.glob("model_checkpoints/*.model")[-1])
except IndexError:
    pass

train(network, tr_load, va_load)
