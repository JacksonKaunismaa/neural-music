import librosa
import librosa.display
import numpy as np
import torch
#import pandas as pd
from torch.utils.data import Dataset
import torchaudio

def encode_classes(classes):
    class_names = set(classes) # gets all unique class types
    class_map = {name:i for i,name in enumerate(class_names)}
    return [class_map[name] for name in classes]

class DatasetConfig:
    win_sz = 6 # num seconds
    sr = 22050  # sample rate
    stft_win_sz = 256*6
    stft_hop_sz = 1024
    num_mels = 128
    batch_sz = 8
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class MusicDataset(Dataset):
    def __init__(self, annotations, config): #sample_size, sr, win_length, hop_length, n_mels):
        #df = pd.read_csv(fname)
        #df["classes"] = encode_classes(df["classes"])
        self.annotations = annotations
        self.win_sz = config.win_sz  # length of each sample
        self.sr = config.sr    # sample rate
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=config.stft_win_sz,
            hop_length=config.stft_hop_sz,
            n_mels=config.num_mels,
        )
        self.gpu = torch.device("cuda:0")
        self.num_mels = config.num_mels
        self.config = config

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        #length = int(np.ceil(self.annotations[idx,2]))
        path = self.annotations[idx,1]
        curr_sr = self.annotations[idx,3]
        frames = torch.tensor(librosa.load(path, offset=self.win_sz, duration=self.win_sz*self.config.bs, sr=curr_sr)[0], device="cpu")
        resamp_frames = torch.reshape(torchaudio.functional.resample(frames, curr_sr, self.sr), [self.config.bs, -1]).to(self.gpu)
        spectrogram = self.mel_extractor(resamp)
        #print("unshifted spec",spectrogram.shape)

        return spectrogram[:,:-1], spectrogram[:,1:], torch.tensor(self.annotations[idx,0])  #x, y, cond
        #return frames, frames, self.annotations[idx,0]  #x, y, cond
