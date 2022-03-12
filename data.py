import librosa
import librosa.display
import numpy as np
import torchlibrosa as tl
import torch
#import pandas as pd
from torch.utils.data import Dataset

def encode_classes(classes):
    class_names = set(classes) # gets all unique class types
    class_map = {name:i for i,name in enumerate(class_names)}
    return [class_map[name] for name in classes]

class MusicDataset(Dataset):
    def __init__(self, annotations, win_size, sr, win_length, hop_length, n_mels):
        #df = pd.read_csv(fname)
        #df["classes"] = encode_classes(df["classes"])
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
        length = int(np.ceil(self.annotations[idx,2]))
        path = self.annotations[idx,1]
        frames = np.array([librosa.load(path, offset=t, duration=self.win_sz, sr=self.sr)[0]
                           for t in range(0,length,self.win_sz)])
        return frames[:,:-1], frames[:,1:], self.annotations[idx,0]  #x, y, cond
