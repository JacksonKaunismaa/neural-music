# TAKEN FROM https://github.com/resemble-ai/MelNet/blob/master/model.py
# training loop and data processing were done by us

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, num_mels, time_steps, n_layers):
        super().__init__()
        # Input layers
        self.freq_extract = nn.GRU(time_steps, time_steps, batch_first=True, bidirectional=True)
        self.time_extract = nn.GRU(num_mels, num_mels, batch_first=True, bidirectional=True)

    def forward(self, spectrogram):
        # Shift the inputs left for time-delay inputs
        # spectrogram: (batch_size, time, freq)

        N,T,F = spectrogram.size()
        freq_input = spectrogram.transpose(1, 2).contiguous()

        freq_features, _ = self.freq_extract(freq_input)   # (batch, freq, 2*time_steps)
        time_features, _ = self.time_extract(spectrogram)  # (batch, time, 2*freq_steps)

        freq_features = freq_features.transpose(1,2).contiguous().view(-1, T, 2*F)

        return torch.cat((time_features, freq_features), 1)


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
        # Print model size
        #self.num_params()

    def forward(self, x, cond, noise):
        # Shift the inputs left for time-delay inputs
        x_time = F.pad(x, [0, 0, -1, 1, 0, 0]).unsqueeze(-1)
        # Shift the inputs down for freq-delay inputs
        x_freq = F.pad(x, [0, 0, 0, 0, -1, 1]).unsqueeze(-1)

        cond = cond.unsqueeze(-1) # add dimension of 1 to be able to do the linear layer

        # Initial transform from 1 to dims
        x_time = self.time_input(x_time) + self.time_cond(cond)
        x_freq = self.freq_input(x_freq) + self.freq_cond(cond)

        # Run through the layers
        x = (x_time, x_freq)
        x_time, x_freq = self.layers(x)

        # Get the mixture params
        x = torch.cat([x_time, x_freq], dim=-1)
        mu, sigma, pi = torch.split(self.fc_out(x), self.n_mixtures, axis=-1)

        sigma = torch.exp(sigma)
        pi = torch.softmax(dim=-1)
        return mu, sigma, pi

    def forward_sample(self, x, cond, noise):
        mu, sigma, pi = self.forward(x, cond, noise)
        mixture = torch.argmax(pi, dim=-1)  # max sampling probably bad because gradients are 0
        return mu[mixture] + sigma[mixture]*noise

class MelNet(nn.Module):
    def __init__(self, dims, layer_sizes, num_classes, num_mels, time_steps, directions, n_mixtures=10):
        super().__init__()
        self.class_embeds = nn.Embedding(num_classes, 1) # sus
        self.tiers = [MelNetTier(dims, n_layers) for n_layers in layer_sizes]

        M, T = num_mels, time_steps
        self.feature_tier = []
        for d in directions:
            if d == 1:
                T //= 2
            else:
                M //= 2
            self.feature_tier.append(FeatureExtraction(M,T,2))

        #   def __init__(self, num_mels, time_steps, n_layers, n_mixtures=10):
        # Print model size
        self.directions = directions
        self.num_params()

    def split(self, x, dim):
        even = torch.arange(x.shape[dim]//2)*2
        odd = even + 1
        if dim == 1:
            return x[:, even, :], x[:, odd, :]
        else:
            return x[:, :, even], x[:, :, odd]

    def interleave(self, x1, x2, dim):
        interleaved = torch.repeat_interleave(x1, 2, dim=dim)
        indices = 1+torch.arange(x1.shape[dim]//2)*2
        if dim == 1:
            interleaved[:,indices,:] = x2
        else:
            interleaved[:,:,indices] = x2
        return interleaved

    def forward(self, x, cond, noise):
        def multi_scale(x, g):
            if g == 0:
                return self.tiers[0].forward_sample(x, self.class_embeds(cond), noise)
            else:
                dim = self.directions[g]
                x_g, x_g_prev = self.split(x, dim)
                x_pred_prev = multi_scale(x_g_prev, g-1)
                prev_features = self.feature_tier[g](x_pred_prev)
                if g == len(self.tiers)-1:
                    return self.tiers[g].forward(x_g, prev_features, noise)
                else:
                    x_pred = self.tiers[g].forward_sample(x_g, prev_features, noise)
                    return self.interleave(x_pred, x_pred_prev, dim)
        return multi_scale(x, len(self.tiers)-1)

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)

def main():
    batchsize = 4
    timesteps = 100
    num_mels = 256
    dims = 256

    #def __init__(self, dims, layer_sizes, num_classes, n_mixtures=10):
    model = MelNet(dims, [4,3,2], 2, num_mels, timesteps, [1,2,2]) # split on freq(highest tier), then freq(mid tier), then time(unconditional tier)

    x = torch.ones(batchsize, timesteps, num_mels)
    z = torch.ones((1), dtype=torch.int64)

    print("Input Shape:", x.shape)
    noise = torch.normal(mean=torch.zeros(batchsize))
    y = model(x, z, noise)

    print("Output Shape", y.shape)

if __name__ == "__main__":
    main()
