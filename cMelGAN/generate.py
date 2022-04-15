import torch
import glob
import numpy as np
import warnings
import librosa
import soundfile
import librosa.display
import matplotlib.pyplot as plt

def invert_and_save(spect, genre, config):
    final_spectrogram = (spect.squeeze()).detach().cpu().numpy()
    db = librosa.power_to_db(final_spectrogram, ref=np.max)
    librosa.display.specshow(db, x_axis="time", y_axis="mel", sr=config.sr, fmax=8000)
    audio = librosa.feature.inverse.mel_to_audio(final_spectrogram, sr=config.sr, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz)
    name = len(glob.glob(f"{genre}_samples/*.wav"))+100
    soundfile.write(f"{genre}_samples/{name}.wav",  audio, config.sr, "PCM_24")
    plt.savefig(f"{genre}_samples/{name}.png")

def test_mel_params(config):
    audio = librosa.load("./music2.wav", duration=config.win_sz, sr=config.sr)[0]
    s = librosa.feature.melspectrogram(audio, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz, sr=config.sr, n_mels=config.num_mels)
    audio = librosa.feature.inverse.mel_to_audio(s, sr=config.sr, hop_length=config.stft_hop_sz, win_length=config.stft_win_sz)
    soundfile.write("test2.wav",  audio, config.sr, "PCM_24")
    print(s.shape)

def gen_one(genre, net_g, config, dev):
    noise = torch.normal(mean=torch.zeros(1, net_g.noise_sz)).to(dev)
    conds = torch.tensor(genre).unsqueeze(0).to(dev)
    current = net_g(noise, conds)[0]
    invert_and_save(current, genre, config)
