# Music generation using MelSpectrogram
### A PyTorch implementation of MelNet https://arxiv.org/pdf/1906.01083.pdf

Spectrograms are like pictures of sounds. They provide a 2-D representation of audio by taking the Fourier transform of the waveform at regular intervals. These intervals, each of which generates a column of the image (Figure 1), represent a far greater time difference compared to the time difference associated with adjacent elements in a waveform representation.

<img width="378" alt="Screen Shot 2022-03-13 at 8 16 21 PM" src="https://user-images.githubusercontent.com/57376402/158085731-f15047ea-f4cd-4d7e-b08e-950f96935107.png">

*figure 1 - MelSpectrogram representation of audio*

## Requirements
PyTorch 

librosa must be installed
https://github.com/librosa/librosa

## Training the network
- In ' data.py  ' we made a datapipeline to automatically load ' wav ' form data and convert them to spectrogram 

## Generating music 
