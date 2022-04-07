# Adapted from https://github.com/seungwonpark/melgan/

import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import torchaudio
#import torch.utils.checkpoint as checkpoint
#ckpt = checkpoint.checkpoint

#def chkpt_fwd(module):  # standard boilerplate code for buffer checkpointing
#    def custom_fwd(*inputs):
#        return module(*inputs)
#    return custom_fwd

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
            #print("Res:", x.shape)
            x = shortcut(x) + block(x)
        return x


class Generator(nn.Module):
    def __init__(self, tr_conf, data_conf):
        super().__init__()
        # Takes in random noise of size (noise_sz,) output shape will be (num_mels, time_len,
        #self.conf = tr_conf
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
        x = self.linear(x).view(-1, *self.start_shape)
        x = self.lin_act(x)
        for layer in self.layers2:
            x = layer(x)
        x = x.squeeze(1)
        for layer in self.layers1:
            x = layer(x)
        return x

    def save(self, epoch, loss, optim, path="model_checkpoints"):
        torch.save({"epoch": epoch, "model_state": self.state_dict(), "loss": loss, "optim_state": optim.state_dict()},
                   f"{path}/gen-{self.num_mels}-{epoch}-{loss:.2f}.model")

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
            #print(params, prev_params, fact)#prev_sz, next_sz)
            act = nn.LeakyReLU(tr_conf.leaky)
            if i == len(tr_conf.d_factors)-1:
                act = nn.Sigmoid()
            layers2.append(nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(
                    prev_params[0], params[0], params[1],
                    stride=params[2], padding=((next_sz-1)*params[2]+params[1]-prev_sz)//2,
                    groups=params[3])),
                act,
            ))
        self.layers_2d = nn.ModuleList(layers2)
        # (num_mels, num_mels) here should be (num_mels, seq_len)
        self.sz = data_conf.num_mels
        self.embeds = nn.Embedding(data_conf.num_classes, self.sz*self.sz)
        self.num_params()

    def forward(self, x, cond):
        embed = self.embeds(cond).view(-1, self.sz, self.sz)
        x = torch.stack((x, embed), dim=1)
        for layer in self.layers_2d:
            x = layer(x)
        return x

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

def main():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    import train
    import data
    train_conf = train.TrainConfig()
    dataset_conf = data.DatasetConfig(num_classes=3)
    g_model = Generator(train_conf, dataset_conf)
    d_model = Discriminator(train_conf, dataset_conf)

    noise_gen = torch.randn((dataset_conf.batch_sz, train_conf.noise_sz))
    condit = torch.tensor(np.ones(dataset_conf.batch_sz), dtype=torch.int32)
    out = g_model(noise_gen, condit)
    print(out.shape)

    out = d_model(out, condit)
    print(out.shape)
    #batchsize = 2
    #timesteps = 50
    #num_mels = 64
    #dims = 64

    ##def __init__(self, dims, layer_sizes, num_classes, n_mixtures=10):
    ##model = MelNet(dims, [4,3,2,2], 2, num_mels, timesteps, [2,1]) # split on freq (highest tier), split on time(mid tier)

    #x = torch.ones(batchsize, timesteps, num_mels)
    #z = torch.ones((1), dtype=torch.int64)

    ##x1,x2 = MelNet.split(x,1)
    ##x3,x4 = MelNet.split(x,2)
    ##print(x1.shape, x3.shape)
    ##x_freq = MelNet.interleave(x3,x4,2)
    ##x_time = MelNet.interleave(x1,x2,1)
    ##print(x_freq.shape, x_time.shape)
    ##print((x-x_freq).sum(), (x-x_time).sum())

    #print("Input Shape:", x.shape)
    #noise = torch.normal(mean=torch.zeros(batchsize,1,1))
    #y = model(x, z, noise)
    #print("Mu shape", y[0].shape, "Sigma shape", y[1].shape, "Pi shape", y[2].shape)
    #y = model(x, z, noise)
    #print(y)
    #print("Mu shape", y[0].shape, "Sigma shape", y[1].shape, "Pi shape", y[2].shape)
    ##print("Output Shape", y.shape)

if __name__ == "__main__":
    main()

