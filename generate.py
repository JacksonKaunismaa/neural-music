import model
import torch
import glob
import numpy as np
import warnings

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
    network.load(glob.glob("model_checkpoints/*.model")[-1])
except IndexError:
    print("Must train a model first")
    raise

desired_steps = 256
current = torch.zeros((1,1,NUM_MELS))
genre = torch.tensor(0)  # 0 corresponds to classical, 1 is lofi
#for _ in desired_steps:
noise = torch.normal(mean=torch.zeros(1,1,1))
out = network(current, genre, noise)
print(out.shape)
