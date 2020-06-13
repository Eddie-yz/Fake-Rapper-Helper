from Preprocessing.dataloader import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa.display
from RecognitionEncoder import DeepSpeech
from torch.autograd import Variable
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import argparse
from StyleRepresentation.model import SingleExtractor
from copy import deepcopy

speed_volume_perturb=False
spec_augment=False
sr = 16000
audio_config = dict(sample_rate=sr,
                          window_size=.02,
                          window_stride=0.01,
                          window='hamming',
                          noise_dir=None,
                          noise_prob=0.4,
                          noise_levels=(0.0, 0.5))


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def load_model(path):
    print("Loading state from model %s" % path)
    package = torch.load(path, map_location=lambda storage, loc: storage)
    model = DeepSpeech(audio_conf=package['audio_conf'])
    model.load_state_dict(package['state_dict'], strict=False)
    return model


# load model parameters
device = torch.device("cuda")
rEncoder = load_model('./deepspeech/librispeech_pretrained_v2.pth')
rEncoder = rEncoder.to(device)

rEncoder_target = load_model('./deepspeech/librispeech_pretrained_v2.pth').to(device)

styEncoder = SingleExtractor(conv_channels=128,
                             sample_rate=16000,
                             n_fft=513,
                             n_harmonic=6,
                             semitone_scale=2,
                             learn_bw='only_Q').to(device)

styEncoder_target = SingleExtractor(conv_channels=128,
                             sample_rate=16000,
                             n_fft=513,
                             n_harmonic=6,
                             semitone_scale=2,
                             learn_bw='only_Q').to(device)

parser = argparse.ArgumentParser()
parser.add_argument('-styleMod', type=str, required=True, help='loss type')
parser.add_argument('-styleInp', type=str, required=True, help='loss type')
parser.add_argument('-l1', type=float, required=True, help='loss type')
parser.add_argument('-l2', type=float, required=True, help='loss type')
parser.add_argument('-inp', type=str, required=True, help='loss type')
parser.add_argument('-out', type=str, required=True, help='loss type')

style_model = parser.parse_args().styleMod
style_name = parser.parse_args().styleInp
filename = parser.parse_args().inp
outname = parser.parse_args().out
lambda1 = parser.parse_args().l1
lambda2 = parser.parse_args().l2

styEncoder.load_state_dict(torch.load(style_model))
styEncoder_target.load_state_dict(torch.load(style_model))

# filename='/home/nio/folder/hip-pop/MC Shan - The Bridge my-free-mp3s.com .wav'
audio1, sample_rate = librosa.load(filename, sr=16000)
D1 = librosa.stft(audio1, n_fft=320)
spec1, phase = librosa.magphase(D1)
spec1 = np.log1p(spec1)

np.save(outname + "_data", spec1)

audio2, sample_rate = librosa.load(style_name, sr=16000)
D2 = librosa.stft(audio2, n_fft=320)
spec2, phase = librosa.magphase(D2)
spec2 = np.log1p(spec2)

criterion = MSELoss()
speech = Variable(torch.FloatTensor(spec1).unsqueeze(0).to(device))
result = Variable(torch.FloatTensor(deepcopy(spec1)).unsqueeze(0).to(device), requires_grad=True)
ref = Variable(torch.FloatTensor(spec2).unsqueeze(0).to(device))

optimizer = Adam([result,], lr=1e-3, betas=(0.9, 0.999))
length = torch.LongTensor([speech.shape[2]])
# length = [speech.shape[2]]
content2 = rEncoder_target(speech.unsqueeze(0), length)
style2 = styEncoder_target(ref)

for i in range(100):
    optimizer.zero_grad()
    loss = 0.
    content1 = rEncoder(result.unsqueeze(0), length)
    loss_content = lambda1 * criterion(content1, content2)

    loss_content.backward(retain_graph=True)
    optimizer.step()

    # loss += lambda1 * loss_content
    # optimizer.zero_grad()
    # style1 = styEncoder(result)
    # loss_style = lambda2 * criterion(style1, style2)
    # # loss += lambda2 * loss_style
    # loss_style.backward(retain_graph=True)
    # # loss.backward(retain_graph=True)
    # optimizer.step()

    if i % 10 == 0:
        # print('iter %d, content loss %.3f, style loss %.3f' % (i, loss_content.item(), loss_style.item()))
        print('iter %d, content loss %.3f' % (i, loss_content.item()))


np.save(outname, result.cpu().detach().numpy())
print(result.cpu().detach().numpy()[0].shape)
# Return the all-zero vector with the same shape of `a_content`
a = np.exp(result.cpu().detach().numpy()[0]) - 1
p = 2 * np.pi * np.random.random_sample(result.shape) - np.pi
n_fft = int(audio_config['sample_rate'] * audio_config['window_size'])

for i in range(50):
    S = a * np.exp(1j * p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, n_fft))

# np.save(outname, x)
librosa.output.write_wav(outname, x, sr)
