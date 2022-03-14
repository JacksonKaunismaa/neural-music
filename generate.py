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
torch.set_default_tensor_type("torch.cuda.FloatTensor")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Model hyperparameters
EPOCHS = 30
DIMS = 64
N_LAYERS = [4,2,2]
DIRECTIONS = [2]
LEARN_RATE = 1e-5
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1

# Data hyperparameters
WIN_SIZE = 6 # num seconds
SR = 11025  # sample rate
STFT_WIN_SIZE = 256
STFT_HOP_SIZE = 1024
NUM_MELS = 128


network = model.MelNet(DIMS, N_LAYERS, 2, NUM_MELS, DIRECTIONS)
try:
    path = "model_checkpoints/422-128-30-67.23719781637192.model" #glob.glob("model_checkpoints/*.model")[-1]
    print("loading", path)
    network.load(path)
except IndexError:
    print("Must train a model first")
    raise

def invert_and_save(spect, genre):
    final_spectrogram = (spect.squeeze()).detach().cpu().numpy()
    db = librosa.power_to_db(final_spectrogram, ref=np.max)
    librosa.display.specshow(db, x_axis="time", y_axis="mel", sr=SR, fmax=8000)
    audio = librosa.feature.inverse.mel_to_audio(final_spectrogram, sr=SR, hop_length=STFT_HOP_SIZE, win_length=STFT_WIN_SIZE)
    name = len(glob.glob(f"{genre}_samples/*.wav"))+100
    #audio = librosa.resample(audio, SR, desired_sr)
    soundfile.write(f"{genre}_samples/{name}.wav",  audio, SR, "PCM_24")
    plt.savefig(f"{genre}_samples/{name}.png")

def test_mel_params():
    audio = librosa.load("./MUSIC_DATA/maestro-v3.0.0/2004/MIDI-Unprocessed_XP_04_R1_2004_03-05_ORIG_MID--AUDIO_04_R1_2004_05_Track05_wav.wav", duration=10, sr=SR)[0]
    s = librosa.feature.melspectrogram(audio, hop_length=STFT_HOP_SIZE, win_length=STFT_WIN_SIZE, sr=SR)
    audio = librosa.feature.inverse.mel_to_audio(s, sr=SR, hop_length=STFT_HOP_SIZE, win_length=STFT_WIN_SIZE)
    soundfile.write("test.wav",  audio, SR, "PCM_24")

def gen_one(genre):
    desired_steps = 256
    desired_sr = 48000
    current = torch.zeros((1,1,NUM_MELS))
    #genre = 0  # 0 corresponds to classical, 1 is lofi
    for _ in tqdm(range(desired_steps)):
        noise = torch.normal(mean=torch.zeros(1,1,1))
        mu,sigma,pi = network(current, torch.tensor(genre), noise)
        categs = Categorical(pi)
        normals = Normal(mu, sigma)
        mixture = MixtureSameFamily(categs, normals)
        next_col = mixture.sample()[0,-1,:].unsqueeze(0).unsqueeze(1)
        current = torch.cat((current, next_col), dim=1)
    invert_and_save(current, genre)

for g in [1,1,1,1,1,1,1,1,1,1,1]:
    gen_one(g)


