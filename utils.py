import librosa
import numpy as np
import torch
import soundfile
import time
from model import *
from packaging import version

def librosa_write(outfile, x, sr):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(outfile, x, sr)
    else:
        soundfile.write(outfile, x, sr)

def wav2spectrum(filename):
    x, sr = librosa.load(filename) #STFT with semi-dB scale
    S = librosa.stft(x, n_fft=N_FFT)
    S = np.log(np.abs(S)+0.1)
    #print(S1, S2)
    #time.sleep(200)
    return S, sr


def spectrum2wav(spectrum, sr, outfile):
    x = librosa.griffinlim(S=np.exp(spectrum)-0.1, n_fft=N_FFT) #Griffin-Lim
    librosa_write(outfile, x, sr)

def compute_content_loss(a_C, a_G):
    m, n_C, n_H, n_W = a_G.shape

    # Reshape a_C and a_G to (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m * n_C, n_H * n_W)
    L_content = torch.square(a_C_unrolled - a_G_unrolled).mean()/4

    return L_content


def gram(A):
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    m, n_C, n_H, n_W = A.shape
    # Reshape a_C and a_G to (m * n_C * n_H, n_W)
    # print(A.shape)
    A_unrolled = A.reshape(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())
    GA = GA / (n_W)
    return GA


def compute_layer_style_loss(a_S, a_G):

    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)
    L_style_layer = torch.square(GS - GG).mean()
    # print(L_style_layer)
    # time.sleep(10)
    return L_style_layer


def compute_variation_loss(a_G):
    m, n_C, n_H, n_W = a_G.shape
    #vectorized, loop-less implementation of variation loss
    loss = torch.sum(torch.pow(a_G[:, :, :-1, :] - a_G[:, :, 1:, :], 2)) + torch.sum(torch.pow(a_G[:, :, :, :-1] - a_G[:, :, :, 1:], 2))
    loss = loss/((n_H - 1)*(n_W - 1))
    return loss