import model
import data
import tqdm
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from torch.autograd import Variable


def train(args_idk_what_yet):
  EPOCHS = 128
  DIMS = 256
  N_LAYERS = 6
  learning_rate = 1e-5

  #criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)
  train_loss_list = np.zeros(EPOCHS)
  valid_loss_list = np.zeros(EPOCHS)
  
  model = MelNet(DIMS, N_LAYERS)
  
  for i in tqdm(range(EPOCHS)):
    
 
