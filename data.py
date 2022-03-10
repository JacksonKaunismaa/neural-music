import librosa
import librosa.display
import numpy as np
import torchlibrosa as tl
import torch
import pandas as pd
from torch.utils.data import Dataset

DATA_DIR = "./MUSIC_DATA/maestro-v3.0.0"
df = pd.read_csv(f"{DATA_DIR}/maestro-v3.0.0.csv")
selection = ["audio_filename", "duration"]  # the only features we care about

train_entries = np.array(df[df.split == "train"][selection])
valid_entries = np.array(df[df.split == "validation"][selection])
test_entries = np.array(df[df.split == "test"][selection])

sample_rate = 44100
win_length = 2048
hop_length = 512
n_mels = 128

class MusicDataset(Dataset):
    def __init__(self, annotations, win_size, sr, win_length=2048, hop_length=512, n_mels=128):
        self.annotations = annotations
        self.win_sz = win_size  # length of each sample
        self.sr = sr
        self.mel_extractor = \
            torch.nn.Sequential( # weird design choice
                tl.Spectrogram(
                    hop_length=hop_length,
                    win_length=win_length,
                ), tl.LogmelFilterBank(
                    sr=sr,
                    n_mels=n_mels,
                    is_log=False, # Default is true
                ))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        length = int(np.ceil(self.annotations[idx,1]))
        path = f"{DATA_DIR}/{self.annotations[idx,0]}"
        frames = np.array([librosa.load(path, offset=t, duration=self.win_sz, sr=self.sr)[0]
                           for t in range(0,length,self.win_sz)])
        return frames
