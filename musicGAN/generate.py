#import model
import torch
import glob
import numpy as np
import warnings
#from torch.distributions import *
#from tqdm import tqdm
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
    #audio = librosa.resample(audio, SR, desired_sr)
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


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    #network = model.MelNet(DIMS, N_LAYERS, 2, NUM_MELS, DIRECTIONS)
    #try:
    #    path = "model_checkpoints/422-128-30-67.23719781637192.model" #glob.glob("model_checkpoints/*.model")[-1]
    #    print("loading", path)
    #    network.load(path)
    #except IndexError:
    #    print("Must train a model first")
    #    raise
    #for g in [1,1,1,1,1,1,1,1,1,1,1]:
    #    gen_one(g)
    import data
    conf = data.DatasetConfig(num_mels=256, stft_hop_sz=864, stft_win_sz=256*8, win_sz=10)
    test_mel_params(conf)
