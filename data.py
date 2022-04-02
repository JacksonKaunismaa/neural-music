import librosa
import librosa.display
#import numpy as np
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
        self.gpu = torch.device("cuda:0")
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=config.stft_win_sz,
            hop_length=config.stft_hop_sz,
            n_mels=config.num_mels,
        ).to(self.gpu)
        self.num_mels = config.num_mels
        self.config = config
        #print(set(annotations[:,3]))
        self.resamplers = {f:torchaudio.transforms.Resample(orig_freq=f, new_freq=self.sr) for f in set(annotations[:,3])}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        #length = int(np.ceil(self.annotations[idx,2]))
        path = self.annotations[idx,1]
        curr_sr = self.annotations[idx,3]
        resampler = self.resamplers[curr_sr]
        frames = torch.tensor(librosa.load(path, offset=0.0, duration=self.win_sz*self.config.batch_sz, sr=curr_sr)[0], device="cpu")
        print(path)
        print(frames.shape)
        truncated = len(frames)-len(frames)%8
        print(truncated)
        frames = torch.reshape(frames[:truncated], [self.config.batch_sz, -1])
        resamp_frames = resampler(frames).to(self.gpu)
        spectrogram = self.mel_extractor(resamp_frames)
        #print("unshifted spec",spectrogram.shape)

        return spectrogram[:,:-1], spectrogram[:,1:], torch.tensor(self.annotations[idx,0])  #x, y, cond
        #return frames, frames, self.annotations[idx,0]  #x, y, cond
