#   cMelGAN
## conditional generative model based on MelSpectrograms, a faster model for limited hardware computing resources
### Last Updated: Apr.13.2022

### First part of the code : A PyTorch implementation of MelNet https://arxiv.org/pdf/1906.01083.pdf
### We propose cMelGAN, inspired by MelGAN https://arxiv.org/abs/1910.06711 and conditional GAN https://arxiv.org/abs/1411.1784


Spectrograms are like pictures of sounds. They provide a 2-D representation of audio by taking the Fourier transform of the waveform at regular intervals. These intervals, each of which generates a column of the image (Figure 1), represent a far greater time difference compared to the time difference associated with adjacent elements in a waveform representation.

<img width="378" alt="Screen Shot 2022-03-13 at 8 16 21 PM" src="https://user-images.githubusercontent.com/57376402/158085731-f15047ea-f4cd-4d7e-b08e-950f96935107.png">

*figure 1 - MelSpectrogram representation of audio*

## cMelGAN Architecture: Generator and Discriminator 
<img width="547" alt="Screen Shot 2022-04-14 at 10 01 55 PM" src="https://user-images.githubusercontent.com/57376402/163506067-660fdc4b-d606-4c8c-8a52-7ab92811f842.png">
*figure 2 - cMelGAN Architecture*

## Requirements
- The model was trained and tested with Python 3.9.7 PyTorch 1.10.2.

- librosa must be installed https://github.com/librosa/librosa. Ther librosa version we used is 0.9.1

## Training the network
- In `data.py` we made a datapipeline to automatically load `.wav` form data and convert them to spectrogram and store them in `.csv` format.
- load data and change the parameters in `train.py` to train the model.

## Generating music 
- run `generate.py` to get some results

## A detailed anlysis of MelNet and our implementations 
- link to paper

## Next Step
-  Try to fix some memory issues to the best we can
-  Look into implementing models like MelGAN, VocGAN, MelGlow, or ParallelWaveGAN, many of which are based off of MelNet
## License 
MIT License
