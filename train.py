import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from utils import *
from model import *
import time
import math
import argparse
cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument('-content', help='Content input')
parser.add_argument('-content_weight', help='Content weight. Default is 1e2', default = 1e2)
parser.add_argument('-style', help='Style input')
parser.add_argument('-style_weight', help='Style weight. Default is 1', default = 1e7)
parser.add_argument('-variation_weight', help='Variation weight. Default is 1', default = 1e-2)
parser.add_argument('-epochs', type=int, help='Number of epoch iterations. Default is 20000', default = 20000)
parser.add_argument('-print_interval', type=int, help='Number of epoch iterations between printing losses', default = 100)
parser.add_argument('-plot_interval', type=int, help='Number of epoch iterations between plot points', default = 100)
parser.add_argument('-audio_interval', type=int, help='Number of epoch iterations between plot points', default = 500)
parser.add_argument('-learning_rate', type=float, default = 0.05)
parser.add_argument('-output', help='Output file name. Default is "output"', default = 'output')
args = parser.parse_args()


CONTENT_FILENAME = args.content
STYLE_FILENAME = args.style

a_content, sr = wav2spectrum(CONTENT_FILENAME)
a_style, sr = wav2spectrum(STYLE_FILENAME)
plt.close("all")
plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Content Spectrum")
plt.imsave('output/Content_Spectrum.png', a_content[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Style Spectrum")
plt.imsave('output/Style_Spectrum.png', a_style[:400, :])

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(a_content_torch.shape)
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
if cuda:
    a_style_torch = a_style_torch.cuda()
print(a_style_torch.shape)
x = torch.rand(5, 3)
print(torch.cuda.is_available())
print("Cuda version: ", torch.__version__)
model_content = RandomCNNContent()
model_content.eval()
model_style = RandomCNNStyle()
model_style.eval()
a_C_var = Variable(a_content_torch, requires_grad=False).float()
a_S_var = Variable(a_style_torch, requires_grad=False).float()
if cuda:
    model_content = model_content.cuda()
    model_style = model_style.cuda()
    a_C_var = a_C_var.cuda()
    a_S_var = a_S_var.cuda()

a_C = model_content(a_C_var)
a_S = model_style(a_S_var)
spectrum2wav(a_content, sr, "output/content.wav")
# Optimizer
learning_rate = args.learning_rate
# a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-1)
a_G_var = torch.clone(a_C_var)
if cuda:
    print("Cuda enabled")
    a_G_var = a_G_var.cuda()
a_G_var.requires_grad = True
optimizer = torch.optim.Adam([a_G_var], lr=learning_rate)

# coefficient of content and style
style_param = args.style_weight
content_param = args.content_weight
variation_param = args.variation_weight

num_epochs = args.epochs
print_every = args.print_interval
plot_every = args.plot_interval
audio_every = args.audio_interval

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
# Train the Model
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    a_G_C = model_content(a_G_var)
    a_G_S = model_style(a_G_var)
    content_loss = content_param * compute_content_loss(a_C, a_G_C)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G_S)
    variation_loss = variation_param * compute_variation_loss(a_G_var)
    min_loss = torch.min(a_G_var)
    loss = style_loss + variation_loss - min_loss
    loss.backward(retain_graph=True)
    optimizer.step()

    # print
    if epoch % print_every == 0:
        print("{} {}% {} content_loss:{:4f} style_loss:{:4f} variation_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                      round(epoch / num_epochs * 100, 1),
                                                                                      timeSince(start),
                                                                                      content_loss.item(),
                                                                                      style_loss.item(),
                                                                                      variation_loss.item(),
                                                                                      loss.item()))
        current_loss += loss.item()

    if epoch % audio_every == 0:
        matplotlib.pyplot.close()
        gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
        gen_audio_C = "output/" + args.output + ".wav"
        spectrum2wav(gen_spectrum, sr, gen_audio_C)
        plt.figure(figsize=(5, 5))
        # we then use the 2nd column.
        plt.subplot(1, 1, 1)
        plt.title("CNN Voice Transfer Result")
        plt.imsave('output/Gen_Spectrum.png', gen_spectrum[:400, :])

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(all_losses)
ax1.set_yscale("log")
plt.savefig('output/loss_curve.png')