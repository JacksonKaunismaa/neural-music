import model
import torch
import glob
import numpy as np
import warnings
from torch.distributions import *
from tqdm import tqdm
import librosa
import soundfile
import librosa.display
import matplotlib.pyplot as plt

def invert_and_save(spect, genre, config):
    final_spectrogram = (spect.squeeze()).detach().cpu().numpy()
    db = librosa.power_to_db(final_spectrogram, ref=np.max)
    librosa.display.specshow(db, x_axis="time", y_axis="mel", sr=config.sr, fmax=8000)
    audio = librosa.feature.inverse.mel_to_audio(final_spectrogram, sr=config.sr, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz)
    name = len(glob.glob(f"{genre.item()}_samples/*.wav"))+100
    soundfile.write(f"{genre.item()}_samples/{name}.wav",  audio, config.sr, "PCM_24")
    plt.savefig(f"{genre.item()}_samples/{name}.png")

def test_mel_params(config):
    audio = librosa.load("./music4.opus", offset=3, duration=config.win_sz, sr=config.sr)[0]
    s = librosa.feature.melspectrogram(audio, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz, sr=config.sr, n_mels=config.num_mels)
    audio = librosa.feature.inverse.mel_to_audio(s, sr=config.sr, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz)
    soundfile.write("test4.wav",  audio, config.sr, "PCM_24")
    print(s.shape)

def gen_one(genre, net, config, dev):
    desired_steps = 256
    genre = torch.tensor(genre,dtype=torch.int32).unsqueeze(0).to(dev)
    current = torch.zeros((1,2,net.num_mels)).to(dev)
    net.eval()
    for _ in tqdm(range(desired_steps)):
        noise = torch.normal(mean=torch.zeros(1,1,1)).to(dev)
        curr_selected = current
        if current.shape[-2] % 2:
            curr_selected = current[:,1:,:]
        mu,sigma,pi = net(curr_selected, genre, noise)
        categs = Categorical(pi)
        normals = Normal(mu, sigma)
        mixture = MixtureSameFamily(categs, normals)
        next_col = mixture.sample()[0,-1,:].unsqueeze(0).unsqueeze(1)
        current = torch.cat((current, next_col), dim=1)
    invert_and_save(current, genre, config)
