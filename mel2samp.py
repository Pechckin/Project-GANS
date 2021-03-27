import torch
import torch.utils.data

import random
from scipy.io import wavfile

import sys
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT
from torch.autograd import Variable

class Mel2Samp(torch.utils.data.Dataset):

    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        
        self.MAX_WAV_VALUE = 32768.0
        self.segment_length, self.sampling_rate = segment_length, sampling_rate
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin,
                                 mel_fmax=mel_fmax)
        
        self.skip = True if 'test' in training_files else False
        with open(training_files, 'r') as f:
             self.data = f.read().splitlines()
        

    def __getitem__(self, index):
        filename = self.data[index]
        sr, audio = wavfile.read(filename)
        audio = torch.from_numpy(audio).float()
        audio_size = audio.shape[0]
        if not self.skip:
            if audio_size < self.segment_length:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio_size, 'constant').data)
            else:
                max_audio_start = audio_size - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start + self.segment_length]
            
        audio = audio / self.MAX_WAV_VALUE
                                            
        audio_norm = Variable(audio.unsqueeze(0), requires_grad=False)
        mel_spec = self.stft.mel_spectrogram(audio_norm)
        mel_spec = torch.squeeze(mel_spec, 0)
                                            
        return (mel_spec, audio)

    def __len__(self):
        return len(self.data)