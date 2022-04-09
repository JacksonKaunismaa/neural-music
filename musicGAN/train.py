#import model
#import data
from tqdm import tqdm
import torch
#import torch.nn as nn
#import matplotlib
#import glob
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import numpy.random as npr
#from torch.autograd import Variable
##from torch.utils.data import DataLoader#import pylab

class TrainConfig:
    # Model hyperparameters
    leaky = 0.2
    start_filts = 256
    start_shape = 4
    noise_sz = 100
    factors = [4, 4, 2, 2] # factors by which the spectrogram should be upscaled
    layers_2d = [[128, 4, 4],   # each tuple is (output_filters, kernel_size, stride)
                 [64, 4, 4],
                 [16, 8, 2],
                 [1, 8, 2]
                ]
    layers_1d = [[4, 2],   # each tuple is (kernel_size, stride)
                 [8, 2],
                 [24, 2],
                 [24, 2],
                 [30, 2],
                ]

    d_layers_2d = [[64, 21, 1, 1], # each is (ouput_filts, kern_sz, stride, groups
                   [128, 22, 4, 4],
                   [256, 22, 4, 16],
                   [512, 22, 4, 32],
                   [1024, 22, 4, 64],
                   [2048, 6, 2, 256]]
    d_factors = [1, 1, 4, 4, 4, 4]
    real_lbl = 1.0
    epochs = 30
    learn_rate = 1e-4
    tr_pct = 0.8
    va_pct = 0.9
    discrim_steps = 1
    batch_sz = 8
    save_every = 10
    dropout = 0.5
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def preprocess(network, samples, device):
    samples = samples.to(device)
    mel_spect = network.mel_extractor(samples)
    noise = torch.normal(mean=torch.zeros(mel_spect.shape[0], network.noise_sz)).to(device)
    return mel_spect.transpose(1,2).contiguous(), noise

def plot_losses(gen_losses, dis_losses):
    ax = plt.subplot(121)
    ax.plot(gen_losses)
    ax.set_title("Generator Loss")
    ax = plt.subplot(122)
    ax.plot(dis_losses)
    ax.set_title("Discriminator Loss")
    plt.plot()

def train(net_g, net_d, opt_g, opt_d, tr_data, va_data, config, device, path="model_checkpoints"):
    gen_losses = []
    dis_losses = []
    va_losses  = []
    va_acc = []
    last_save = -np.inf
    #torch.backends.cudnn.benchmark = True # apparently this makes it faster if consistent batch size
    for epoch in range(config.epochs):
        net_g.train() # set to training mode
        net_d.train()
        for samples,cond in tqdm(tr_data):
            real_mel, noise = preprocess(net_g, samples, device)
            cond = cond.to(device)
            # update generator
            opt_g.zero_grad()
            fake_mel = net_g(noise, cond)
            disc_fake = net_d(fake_mel, cond)
            loss_g = torch.pow(disc_fake - config.real_lbl, 2).mean()
            loss_g.backward()
            opt_g.step()
            opt_g.zero_grad()
            gen_losses.append(loss_g.item())

            # update discriminator
            loss_total_d = 0.0
            for _ in range(config.discrim_steps):
                opt_d.zero_grad()
                loss_d = 0.0
                fake_mel = net_g(noise, cond)
                discrim_fake = net_d(fake_mel, cond)
                discrim_real = net_d(real_mel, cond)
                # using least squares loss
                loss_d += torch.pow(discrim_real - config.real_lbl, 2).mean()
                loss_d += torch.pow(discrim_fake - (1-config.real_lbl), 2).mean()
                loss_d.backward()
                opt_d.step()
                loss_total_d += loss_d.item()
            dis_losses.append(loss_total_d)

        correct_discrim = 0
        va_loss_d = 0.0
        va_loss_g = 0.0
        net_g.eval()
        net_d.eval()
        for samples, cond in tqdm(va_data):
            real_mel, noise = preprocess(net_g, samples, device)
            cond = cond.to(device)
            # discriminator accuracy, generate equal numbers of fake and real
            fake_mel = net_g(noise, cond)
            discrim_fake = net_d(fake_mel, cond)
            discrim_real = net_d(real_mel, cond)
            # using least squares loss
            va_loss_d += ((discrim_real - config.real_lbl)**2).mean().item()
            va_loss_d += ((discrim_fake - (1-config.real_lbl))**2).mean().item()
            va_loss_g += torch.pow(disc_fake - (1-config.real_lbl), 2).mean()
            correct_discrim += torch.abs(torch.round(discrim_fake)-(1-config.real_lbl)).sum().item()
            correct_discrim += torch.abs(torch.round(discrim_real)-config.real_lbl).sum().item()
        va_losses.append(va_loss_d)
        va_acc.append(100.*correct_discrim/(2*len(va_data)*config.batch_sz))
        print(f"va_loss: {va_losses[-1]}, va_acc: {va_acc[-1]}")
        if epoch - last_save > config.save_every:  # save every 5 epochs
            print("Saving network at epoch", epoch)
            net_g.save(epoch, va_losses[-1], opt_g, path=path)
            net_d.save(epoch, va_losses[-1], opt_d, path=path)
            last_save = epoch
    return gen_losses, dis_losses, va_losses, va_acc
