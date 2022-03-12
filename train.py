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

def gaussian_pdf(x, mu, sigmasq):
  # NOTE: we could use the new `torch.distributions` package for this now
  return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2)


def loss_fn(pi, sigmasq, mu, target):
  # PRML eq. 5.153, p. 275
  # compute the likelihood p(y|x) by marginalizing p(z)p(y|x,z)
  # over z. for now, we assume the prior p(w) is equal to 1,
  # although we could also include it here.  to implement this,
  # we average over all examples of the negative log of the sum
  # over all K mixtures of p(z)p(y|x,z), assuming Gaussian
  # distributions.  here, p(z) is the prior over z, and p(y|x,z)
  # is the likelihood conditioned on z and x.
  losses = Variable(torch.zeros(n))  # p(y|x)
  for i in range(k):  # marginalize over z
    likelihood_z_x = gaussian_pdf(target, mu[:, i*t:(i+1)*t], sigmasq[:, i])
    prior_z = pi[:, i]
    losses += prior_z * likelihood_z_x
  loss = torch.mean(-torch.log(losses))
  return loss


def train(args_idk_what_yet):
  EPOCHS = 128
  DIMS = 256
  N_LAYERS = 6
  learning_rate = 1e-5
  params = [*idk_what_yet*]

  optimizer = torch.optim.Adam(params,lr=learning_rate)
  train_loss_list = np.zeros(EPOCHS)
  #valid_loss_list = np.zeros(EPOCHS)
  
  model = MelNet(DIMS, N_LAYERS)
  
  for epoch in tqdm(range(EPOCHS)):
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    
    train_loss_list[epoch] = loss
    
return *some_parameteres*
    
 
