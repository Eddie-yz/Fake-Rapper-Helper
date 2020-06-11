from scipy.io.wavfile import read
import librosa
import numpy as np
from numpy import random
from numpy.random import randint
import os
from os import listdir
from os import path
from torch.utils.data import Dataset
    
# deepspeech   
def load_audio(path, audiotime):
    # sample_rate, sound = read(path)
    sound, sample_rate = librosa.load(path, sr=16000)
    
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    selection = len(sound)-sample_rate*audiotime
    position = randint(0,selection)
    time = position/sample_rate
    sound = sound[position: position+sample_rate*audiotime ]
    return sound, time

def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y,_ = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio


class SpectrogramParser():
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(SpectrogramParser, self).__init__()
        self.audiotime = 10
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = audio_conf['window']
        self.normalize = normalize
        self.speed_volume_perturb = speed_volume_perturb
        self.spec_augment = spec_augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')
        

    def parse_audio(self, audio_path):
        time = 0
        if self.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y, time = load_audio(audio_path, self.audiotime)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        '''
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        if self.spec_augment:
            spect = spec_augment(spect)
        '''

        return spect, time
        
    # random cnn
    def spectrum2wav(self, spectrum, sr, outfile):
        # Return the all-zero vector with the same shape of `a_content`
        a = np.exp(spectrum) - 1
        p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
        n_fft = int(self.sample_rate * self.window_size)
        for i in range(50):
            S = a * np.exp(1j * p)
            x = librosa.istft(S)
            p = np.angle(librosa.stft(x, n_fft))
        librosa.output.write_wav(outfile, x, sr)
        
        
class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, pos_dir, neg_dir=None,normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.pos_files = os.listdir(pos_dir)
        _sample = np.load(path.join(self.pos_dir,self.pos_files[0]))
        self.sample_size = _sample.shape
        if self.neg_dir:
            self.neg_files = os.listdir(neg_dir)
            self.size = min(len(self.pos_files),len(self.neg_files))
        else:
            self.neg_files = os.listdir(pos_dir)
            self.size = len(self.pos_files)
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)

    def __getitem__(self, index):
        pos_file = path.join(self.pos_dir, self.pos_files[index])
        pos_sample = np.load(pos_file)
        if self.neg_dir:
            neg_file = path.join(self.neg_dir, self.neg_files[index])
            neg_sample = np.load(neg_file)
        else:
            neg_sample = pos_sample
        return pos_sample, neg_sample

    def __len__(self):
        return self.size
    
    def shuffle(self):
        random.shuffle(self.pos_files)
        if self.neg_dir:
            random.shuffle(self.neg_files)