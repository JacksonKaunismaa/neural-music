import librosa
import librosa.display
import numpy as np
import torch
#import pandas as pd
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm
#import time

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
    num_classes = None # will be written to when creating dataset
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class MusicDataset(Dataset):
    def __init__(self, annotations, config): #sample_size, sr, win_length, hop_length, n_mels):
        self.annotations = annotations
        self.win_sz = config.win_sz  # length of each sample
        self.sr = config.sr    # sample rate
        self.num_mels = config.num_mels
        self.config = config
        self.resamplers = {f:torchaudio.transforms.Resample(orig_freq=f, new_freq=self.sr) for f in set(annotations[:,3])}

        self.conds = []
        self.resamps = []
        for i, annotate in tqdm(enumerate(annotations)):
            song_chunks = int(np.floor(annotate[2]//self.win_sz))
            path = annotate[1]
            curr_sr = annotate[3]
            cond = torch.tensor(annotate[0], device="cpu")
            resampler = self.resamplers[curr_sr]
            frames = torch.tensor(librosa.load(path, offset=0., duration=self.win_sz*song_chunks, sr=curr_sr)[0], device="cpu")
            resamp_frames = resampler(frames).reshape(song_chunks, -1)
            self.conds += [cond for _ in range(song_chunks)]
            self.resamps += [r for r in resamp_frames]

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        return self.resamps[idx], self.conds[idx]
