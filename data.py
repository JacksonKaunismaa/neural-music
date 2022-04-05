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
        #self.gpu = torch.device("cuda:0")
        #self.mel_extractor = torchaudio.transforms.MelSpectrogram(
        #    sample_rate=self.sr,
        #    n_fft=config.stft_win_sz,
        #    hop_length=config.stft_hop_sz,
        #    n_mels=config.num_mels,
        #).to(self.gpu)
        self.num_mels = config.num_mels
        self.config = config
        self.resamplers = {f:torchaudio.transforms.Resample(orig_freq=f, new_freq=self.sr) for f in set(annotations[:,3])}

        #self.idx_to_annotation = []
        #self.idx_to_offset = []
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
            #print(curr_sr, librosa.get_samplerate(path))
            self.conds += [cond for _ in range(song_chunks)]
            self.resamps += [r for r in resamp_frames]
            #for offset in range(song_chunks):
            #    #start = time.perf_counter()
            #    #frames = torch.tensor(librosa.load(path, offset=offset*config.win_sz, duration=self.win_sz, sr=curr_sr)[0], device="cpu")
            #    #print("loading", time.perf_counter() - start)
            #    start = time.perf_counter()
            #    resamp_frames = resampler(frames)#.to(self.gpu)
            #    print("resampling", time.perf_counter() - start)
            #    self.conds.append(cond)
            #    self.resamps.append(resamp_frames)

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, idx):
        return self.resamps[idx], self.conds[idx]
