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
    def __init__(self, annotations, sample_size, sr, win_length, hop_length, n_mels):
        #df = pd.read_csv(fname)
        #df["classes"] = encode_classes(df["classes"])
        self.annotations = annotations
        self.win_sz = sample_size  # length of each sample
        self.sr = sr    # sample rate
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
        self.num_mels = n_mels
        self.time_steps = int(np.floor((sample_size*sr-win_length)/hop_length))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        length = int(np.ceil(self.annotations[idx,2]))
        #print(length)
        path = self.annotations[idx,1]
        #print("loading", path)
        #print(self.annotations[idx,0])
        frames = np.array([librosa.load(path, offset=t, duration=self.win_sz, sr=self.sr)[0]
                           for t in range(0,min(length-self.win_sz,8*self.win_sz),self.win_sz)]) # cut off the last segment of the song (could also do padding)
        #print(self.win_sz, length)
        #print(len(frames), [t for t in range(0, length, self.win_sz)])
        #print([(f.dtype,f.shape) for f in frames])
        #print(frames.shape)
        spectrogram = self.mel_extractor(torch.tensor(frames)).squeeze()
        #print("unshifted spec",spectrogram.shape)

        return spectrogram[:,:-1], spectrogram[:,1:], torch.tensor(self.annotations[idx,0])  #x, y, cond
        #return frames, frames, self.annotations[idx,0]  #x, y, cond
