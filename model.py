import torch
import torch.nn as nn
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False

N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32

class RandomCNNContent(nn.Module):
    def __init__(self):
        super(RandomCNNContent, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 7), stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 7), stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 7), stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5), stride=1, padding='same'),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        return out

class RandomCNNStyle(nn.Module):
    def __init__(self):
        super(RandomCNNStyle, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5), stride=1),
            nn.LeakyReLU(0.2)
            # nn.BatchNorm2d(128)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 11), stride=1),
            nn.LeakyReLU(0.2)
            # nn.BatchNorm2d(64)
        )

        
    def forward(self, x):
        out1 = self.encoder1.forward(x)
        # out2 = self.encoder2.forward(out1)
        return out1

"""
a_random = Variable(torch.randn(1, 1, 257, 430)).float()
model = RandomCNN()
a_O = model(a_random)
print(a_O.shape)
"""