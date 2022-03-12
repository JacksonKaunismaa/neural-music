import model
import data
import tqdm
import torch
#import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import numpy.random as npr
from torch.autograd import Variable

EPOCHS = 128
DIMS = 256
N_LAYERS = [8,3,2,2]
DIRECTIONS = [2,2,1]
learning_rate = 1e-5

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


def train(tr_data, va_data):
    network = model.MelNet(DIMS, N_LAYERS, 2, data.num_mels, data.time_steps, DIRECTIONS)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    train_loss_list = []    #np.zeros(EPOCHS)
    valid_loss_list = []    #np.zeros(EPOCHS)
    #model = MelNet(DIMS, N_LAYERS)
    for epoch in tqdm(range(EPOCHS)):
        for x,y,cond in tr_data:
            optimizer.zero_grad()
            noise = torch.normal(size=(x.shape[0],1,1))
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
