import librosa
import numpy as np

sr = 16000

audio1, sample_rate = librosa.load("./n_in_paris_speech.wav", sr=16000)
D1 = librosa.stft(audio1, n_fft=320)
spec1, phase = librosa.magphase(D1)
result = np.log1p(spec1)
np.save("./n_in_paris_speech", result)

audio_config = dict(sample_rate=sr,
                          window_size=.02,
                          window_stride=0.01,
                          window='hamming',
                          noise_dir=None,
                          noise_prob=0.4,
                          noise_levels=(0.0, 0.5))

# result = np.load("./n_in_paris_styled.wav_data.npy")
result = np.load("./n_in_paris_speech.npy")
print(result.shape)
a = np.exp(result) - 1
p = 2 * np.pi * np.random.random_sample(result.shape) - np.pi
n_fft = int(audio_config['sample_rate'] * audio_config['window_size'])

for i in range(50):
    S = a * np.exp(1j * p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, n_fft))

# librosa.output.write_wav("./n_in_paris_styled.wav", x, sr)
librosa.output.write_wav("./n_in_paris_saved.wav", x, sr)
