# PyTorch Implementation of https://arxiv.org/pdf/1906.01083.pdf
# TAKEN FROM https://github.com/resemble-ai/MelNet/blob/master/model.py, with heavy modification
# added FeatureExtraction, MelNetTier, and the overall multi-scale architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import torch.utils.checkpoint as checkpoint
ckpt = checkpoint.checkpoint

def chkpt_fwd(module):  # standard boilerplate code for buffer checkpointing
    def custom_fwd(*inputs):
        return module(*inputs)
    return custom_fwd


class FrequencyDelayedStack(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.rnn = nn.GRU(dims, dims, batch_first=True)

    def forward(self, x_time, x_freq):
        # sum the inputs
        x = x_time + x_freq

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x.size()
        # collapse the first two axes
        x = x.view(-1, M, D)

        # Through the RNN
        x, _ = self.rnn(x)
        return x.view(B, T, M, D)


class TimeDelayedStack(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.bi_freq_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)
        self.time_rnn = nn.GRU(dims, dims, batch_first=True)

    def forward(self, x_time):

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x_time.size()

        # Collapse the first two axes
        time_input = x_time.transpose(1, 2).contiguous().view(-1, T, D)
        freq_input = x_time.view(-1, M, D)

        # Run through the rnns
        x_1, _ = self.time_rnn(time_input)
        x_2_and_3, _ = self.bi_freq_rnn(freq_input)

        # Reshape the first two axes back to original
        x_1 = x_1.view(B, M, T, D).transpose(1, 2)
        x_2_and_3 = x_2_and_3.view(B, T, M, 2 * D)

        # And concatenate for output
        x_time = torch.cat([x_1, x_2_and_3], dim=3)
        return x_time


class Layer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.freq_stack = FrequencyDelayedStack(dims)
        self.freq_out = nn.Linear(dims, dims)
        self.time_stack = TimeDelayedStack(dims)
        self.time_out = nn.Linear(3 * dims, dims)

    def forward(self, x):
        # unpack the input tuple
        x_time, x_freq = x

        # grab a residual for x_time
        x_time_res = x_time
        # run through the time delayed stack
        x_time = self.time_stack(x_time)
        # reshape output
        x_time = self.time_out(x_time)
        # connect time residual
        x_time = x_time + x_time_res

        # grab a residual for x_freq
        x_freq_res = x_freq
        # run through the freq delayed stack
        x_freq = self.freq_stack(x_time, x_freq)
        # reshape output TODO: is this even needed?
        x_freq = self.freq_out(x_freq)
        # connect the freq residual
        x_freq = x_freq + x_freq_res
        return [x_time, x_freq]

class FeatureExtraction(nn.Module):
    def __init__(self, num_mels, n_layers):
        super().__init__()
        # Input layers
        self.time_fwd = nn.GRU(num_mels, num_mels, batch_first=True)
        self.time_back = nn.GRU(num_mels, num_mels, batch_first=True)
        self.weights = nn.Linear(2,1)

    def forward(self, spectrogram):
        # Shift the inputs left for time-delay inputs
        time_fwd_feats, _ = self.time_fwd(spectrogram)
        time_back_feats, _ = self.time_back(spectrogram.flip(1))
        stacked = torch.stack((time_fwd_feats, time_back_feats), dim=-1)#, freq_fwd_feats.transpose(1,2), freq_back_feats.transpose(1,2)), dim=-1)  # completely made up btw
        return self.weights(stacked).squeeze(-1)

class MelNetTier(nn.Module):
    def __init__(self, dims, n_layers, n_mixtures=10):
        super().__init__()
        # Input layers
        self.freq_input = nn.Linear(1, dims)
        self.time_input = nn.Linear(1, dims)

        self.time_cond = nn.Linear(1, dims)
        self.freq_cond = nn.Linear(1, dims)

        # Main layers
        self.layers = nn.Sequential(
            *[Layer(dims) for _ in range(n_layers)]
        )

        # Output layer
        self.fc_out = nn.Linear(2 * dims, 3 * n_mixtures)
        self.n_mixtures = n_mixtures
        self.sampler = True
        self.softmax_pi = torch.nn.Softmax(dim=-1)

    def forward(self, x, cond, noise, sample=True):
        if sample and self.sampler:
            return self.forward_sample(x, cond, noise)
        # Shift the inputs left for time-delay inputs
        x_time = F.pad(x, [0, 0, -1, 1, 0, 0]).unsqueeze(-1)
        # Shift the inputs down for freq-delay inputs
        x_freq = F.pad(x, [0, 0, 0, 0, -1, 1]).unsqueeze(-1)

        cond = cond.unsqueeze(-1) # add dimension of 1 to be able to do the linear layer

        # Initial transform from 1 to dims
        if cond.shape[2] == 1:  # lowest tier, genre embeddings (probably bad way to check)
            shaped_cond = torch.reshape(cond, (-1, 1, 1, cond.shape[-2]))
            x_time = self.time_input(x_time) + shaped_cond  #cond#self.time_cond(cond)
            x_freq = self.freq_input(x_freq) + shaped_cond  #self.freq_cond(cond)
        else:
            x_time = self.time_input(x_time) + self.time_cond(cond)
            x_freq = self.freq_input(x_freq) + self.freq_cond(cond)

        # Run through the layers
        x = (x_time, x_freq)
        x_time, x_freq = self.layers(x)

        # Get the mixture params
        x = torch.cat([x_time, x_freq], dim=-1)
        mu, sigma, pi = torch.split(self.fc_out(x), self.n_mixtures, dim=-1)

        #sigma = torch.exp(sigma)    # causes explosion!
        sigma = torch.exp(sigma/(self.n_mixtures*torch.norm(sigma,dim=-1).unsqueeze(-1))) # scale by n_mixtures (???) at least it further reduces explosion problem
        pi = self.softmax_pi(pi)
        return mu, sigma, pi

    def forward_sample(self, x, cond, noise):
        mu, sigma, pi = self.forward(x, cond, noise, sample=False)
        mu_weighted = (mu*pi).sum(axis=-1)    # probably totally unjustified
        sigma_weighted = (sigma*pi).sum(axis=-1)
        return mu_weighted + sigma_weighted*noise

class MelNet(nn.Module):
    def __init__(self, tr_config, data_config, feature_layers=1, n_mixtures=10):
        super().__init__()
        #assert(len(layer_sizes)-2 == len(directions))
        self.class_embeds = nn.Embedding(data_config.num_classes, tr_config.dims)

        self.tiers = nn.ModuleList([MelNetTier(tr_config.dims, n_layer) for n_layer in tr_config.n_layers])
        self.tiers[-1].sampler = False
        self.g_mid = len(self.tiers)//2

        self.layer_sizes = "".join([str(i) for i in tr_config.n_layers])  # for saving model parameters

        M = data_config.num_mels
        feature_tiers = [FeatureExtraction(M,feature_layers)]
        for d in tr_config.directions[::-1]:
            if d == 2:
                M //= 2
            feature_tiers.append(FeatureExtraction(M,feature_layers))
        self.feature_tier = nn.ModuleList(feature_tiers[::-1])
        self.directions = tr_config.directions
        self.num_mels = data_config.num_mels
        self.dims = tr_config.dims
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=data_config.sr,
            n_fft=data_config.stft_win_sz,
            hop_length=data_config.stft_hop_sz,
            n_mels=data_config.num_mels,
        )
        self.num_params()

    def save(self, epoch, loss, optim, path="model_checkpoints"):
        torch.save({"epoch": epoch, "model_state": self.state_dict(), "loss": loss, "optim_state": optim.state_dict()},
                   f"{path}/{self.layer_sizes}-{self.num_mels}-{self.dims}-{epoch}-{loss}.model")

    def load(self, path, optim):
        model_ckpt = torch.load(path)
        self.load_state_dict(model_ckpt["model_state"])
        if optim:
            optim.load_state_dict(model_ckpt["optim_state"])
        #return model_ckpt["epoch"]  # kinda pointless


    @staticmethod
    def split(x, dim):
        even = torch.arange(x.shape[dim]//2)*2
        odd = even + 1
        if dim == 1:
            return x[:, even, :], x[:, odd, :]
        else:
            return x[:, :, even], x[:, :, odd]

    @staticmethod
    def interleave(x1, x2, dim):
        interleaved = torch.repeat_interleave(x1, 2, dim=dim)
        indices = 1+torch.arange(x1.shape[dim])*2
        if dim == 1:
            interleaved[:,indices,:] = x2
        else:
            interleaved[:,:,indices] = x2
        return interleaved

    def forward(self, x, cond, noise):
        def multi_scale(x, g):
            if g == 0:
                condit_embeds = self.class_embeds(cond)
                return ckpt(chkpt_fwd(self.tiers[0]), x, condit_embeds, noise)
            else:
                dim = self.directions[g-1]
                x_g, x_g_prev = self.split(x, dim)
                x_pred_prev = multi_scale(x_g_prev, g-1)
                prev_features = self.feature_tier[g-1](x_pred_prev)
                x_pred = ckpt(chkpt_fwd(self.tiers[g]), x_g, prev_features, noise)
                return self.interleave(x_pred, x_pred_prev, dim)
        prev_context = multi_scale(x, len(self.tiers)-2)
        prev_features = self.feature_tier[-1](prev_context)
        final_distrib = ckpt(chkpt_fwd(self.tiers[-1]), x, prev_features, noise)
        return final_distrib

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)
