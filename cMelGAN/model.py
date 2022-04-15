# Adapted from https://github.com/seungwonpark/melgan/

import torch
import torch.nn as nn
import numpy as np
import torchaudio

class ResBlock(nn.Module):  # should preserve shape
    # note this is basically completely from https://github.com/seungwonpark/melgan/blob/master/model/res_stack.py
    def __init__(self, channels, conf, dimensionality):
        super().__init__()
        if dimensionality == 2:
            pad = nn.ReflectionPad2d
            conv = nn.Conv2d
        else:
            pad = nn.ReflectionPad1d
            conv = nn.Conv1d

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(conf.leaky),
                pad(3**i),
                nn.utils.weight_norm(conv(channels, channels, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(conf.leaky),
                nn.utils.weight_norm(conv(channels, channels, kernel_size=1)),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([  # is this really necessary?
            nn.utils.weight_norm(conv(channels, channels, kernel_size=1))
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x


class Generator(nn.Module):
    def __init__(self, tr_conf, data_conf):
        super().__init__()
        assert(len(tr_conf.factors) == len(tr_conf.layers_2d))
        layers2 = []
        next_sz = tr_conf.start_shape   # `factors` only matters for num_mels, the time dimension can be anything
        iterator = zip([[tr_conf.start_filts]] + tr_conf.layers_2d, tr_conf.layers_2d, tr_conf.factors)
        for prev_params, params, fact in iterator:
            prev_sz = next_sz
            next_sz = params[2]*(next_sz-1)+params[1]
            layers2.append(nn.Sequential(
                nn.utils.weight_norm(nn.ConvTranspose2d(
                    prev_params[0], params[0], params[1],
                    stride=params[2], padding=(next_sz-prev_sz*fact)//2)),
                nn.LeakyReLU(tr_conf.leaky),
                nn.Dropout(tr_conf.dropout),
                ResBlock(params[0], tr_conf, 2)
            ))
        layers1 = []
        next_sz = data_conf.num_mels # really this should be seq_len
        for params in tr_conf.layers_1d:
            layers1.append(nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(  # add padding to keep sequence length same
                    data_conf.num_mels, data_conf.num_mels, params[0],
                    stride=params[1], padding=(params[1]*(next_sz-1)-next_sz+params[0])//2)),
                nn.LeakyReLU(tr_conf.leaky),
                nn.Dropout(tr_conf.dropout),
                ResBlock(data_conf.num_mels, tr_conf, 1)
            ))
        self.layers1 = nn.ModuleList(layers1)
        self.layers2 = nn.ModuleList(layers2)

        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=data_conf.sr,
            n_fft=data_conf.stft_win_sz,
            hop_length=data_conf.stft_hop_sz,
            n_mels=data_conf.num_mels,
        )

        self.noise_sz = tr_conf.noise_sz
        self.linear = nn.Linear(tr_conf.noise_sz, tr_conf.start_filts*(tr_conf.start_shape*tr_conf.start_shape))
        self.embed = nn.Embedding(data_conf.num_classes, tr_conf.noise_sz)
        self.start_shape = (tr_conf.start_filts,tr_conf.start_shape,tr_conf.start_shape)
        self.lin_act = nn.LeakyReLU(tr_conf.leaky)
        self.num_mels = data_conf.num_mels
        self.num_params()

    def forward(self, x, cond):
        embed = self.embed(cond)
        x = x * embed # elementwise multiply
        x = self.linear(x).view(-1, *self.start_shape).contiguous()
        x = self.lin_act(x)
        for layer in self.layers2:
            x = layer(x)
        x = x.squeeze(1)
        return x

    def save(self, epoch, loss, optim, path="model_checkpoints"):
        torch.save({"epoch": epoch, "model_state": self.state_dict(), "loss": loss, "optim_state": optim.state_dict()},
                   f"{path}/gen-{self.num_mels}-{epoch}-{loss:.3f}.model")

    def load(self, path, optim):
        model_ckpt = torch.load(path)
        self.load_state_dict(model_ckpt["model_state"])
        if optim:
            optim.load_state_dict(model_ckpt["optim_state"])

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Generator Parameters: %.3fM' % parameters)

class Discriminator(nn.Module):
    def __init__(self, tr_conf, data_conf):
        super().__init__()
        layers2 = []    # we start with 2 input channels by (mel_spect, class_embed)
        assert(len(tr_conf.d_factors) == len(tr_conf.d_layers_2d))
        iterator = enumerate(zip([[2]] + tr_conf.d_layers_2d, tr_conf.d_layers_2d, tr_conf.d_factors))
        next_sz = data_conf.num_mels  # really there should be some sort of handling here for non-square spectrograms
        for i, (prev_params, params, fact) in iterator:
            prev_sz = next_sz
            next_sz = prev_sz//fact
            layers2.append(nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(
                    prev_params[0], params[0], params[1],
                    stride=params[2], padding=((next_sz-1)*params[2]+params[1]-prev_sz)//2,
                    groups=params[3])),
                nn.LeakyReLU(tr_conf.leaky),
            ))
        self.layers_2d = nn.ModuleList(layers2)
        # (num_mels, num_mels) here should be (num_mels, seq_len)
        self.sz = data_conf.num_mels
        self.embeds = nn.Embedding(data_conf.num_classes, self.sz*self.sz)
        self.out_feats = tr_conf.d_layers_2d[-1][0]
        self.linear_out = nn.Linear(self.out_feats, 1)
        self.sigm_out = nn.Sigmoid()
        self.num_params()

    def forward(self, x, cond):
        embed = self.embeds(cond).view(-1, self.sz, self.sz).contiguous()
        x = torch.stack((x, embed), dim=1)
        for layer in self.layers_2d:
            x = layer(x)
        lbl = self.sigm_out(self.linear_out(x.view(-1, self.out_feats))).contiguous()
        return lbl

    def save(self, epoch, loss, optim, path="model_checkpoints"):
        torch.save({"epoch": epoch, "model_state": self.state_dict(), "loss": loss, "optim_state": optim.state_dict()},
                   f"{path}/dis-{self.sz}-{epoch}-{loss:.2f}.model")

    def load(self, path, optim):
        model_ckpt = torch.load(path)
        self.load_state_dict(model_ckpt["model_state"])
        if optim:
            optim.load_state_dict(model_ckpt["optim_state"])

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Discriminator Parameters: %.3fM' % parameters)

