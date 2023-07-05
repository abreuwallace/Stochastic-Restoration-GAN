import os
from typing import Tuple
import torchaudio
from torch.utils.data import Dataset
import torch
from torch import Tensor
from pathlib import Path
from torchaudio.transforms import Spectrogram

def load_audio_item(filepath: str, path: str, label_name: str) -> Tuple[Tensor, int, str, str]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    if label_name is not None:
        label = label_name
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, filename



class AudioFolder(Dataset):
    """Create a Dataset from Local Files.

    Args:
        path (str): Path to the directory where the dataset is found or downloaded.
        suffix (str) : Audio file type, defaulted to ".WAV".
        pattern (str) : Find pathnames matching this pattern. Defaulted to "*/*" 
        new_sample_rate (int) : Resample audio to new sample rate specified.
        spectrogram_transform (bool): If `True` transform the audio waveform and returns it  
        transformed into a spectrogram tensor.
        label (str): The label is pulled from the folders in the path, this allows you to statically
        define the label string.
    """


    def __init__(
            self,
            path: str,
            suffix: str = ".wav",
            pattern: str = "*/*/*",
            fs: int = 44100,
            seg_len = 4,
            overlap = 2,
            stft: bool = True,
            win_len: int = 2048, 
            hop_len: int = 512, 
            n_fft: int = 2048,
            complex_as_channels: bool = True,
            label: str = None
        ):
        
        self._path = path
        self._stft = stft
        self._sample_rate = fs
        self._label = label
        self._seg_len = seg_len
        self._overlap = overlap

        walker = sorted(str(p) for p in Path(self._path).glob(f'{pattern}{suffix}'))
        self._walker = list(walker)

        self._total_len = len(self._walker)

        self._original_wave_chunks = torch.Tensor([])
        self._degraded_wave_chunks = torch.Tensor([])
        
        loaded_files = 0

        for filename in self._walker:
            waveform, _ = torchaudio.load(filename)
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
            waveform_chunks = torch.squeeze(waveform_mono.unfold(1, seg_len * self._sample_rate, overlap * self._sample_rate))
            if loaded_files < self._total_len // 2:
                self._original_wave_chunks = torch.cat((self._original_wave_chunks, waveform_chunks), dim=0)
                loaded_files += 1
            else:
                self._degraded_wave_chunks = torch.cat((self._degraded_wave_chunks, waveform_chunks), dim=0)
                loaded_files += 1

        if self._stft:
            self.window_length = win_len  
            self.hop_length = hop_len
            self.window = torch.hann_window(self.window_length)
            self.n_fft = n_fft
            self.complex_as_channels = complex_as_channels
            self.spectrogram = Spectrogram(n_fft=n_fft, win_length=self.window_length, hop_length=self.hop_length,
                                            power=None)
        
    def __getitem__(self, n):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the file to be loaded
        """
        if self._stft:
            Xorig = torch.unsqueeze(torch.view_as_real(self.spectrogram(self._original_wave_chunks[n])), dim=0)
            Xdeg = torch.unsqueeze(torch.view_as_real(self.spectrogram(self._degraded_wave_chunks[n])), dim=0)
            if self.complex_as_channels:
                Ch, Fr, T, _ = Xorig.shape
                Xorig = Xorig.reshape(2 * Ch, Fr, T)
                Xdeg = Xdeg.reshape(2 * Ch, Fr, T)
                return Xdeg, Xorig
        else:
            return torch.unsqueeze(self._degraded_wave_chunks[n], dim=0), torch.unsqueeze(self._original_wave_chunks[n], dim=0)

    def __len__(self) -> int:
        return len(self._walker)
    
    def get_sample_rate(self):
       return self._sample_rate 
    

def main():
    train_data = AudioFolder(path="/home/wallace.abreu/Mestrado/Stochastic-Restoration-GAN/data/", 
                            suffix=".wav",
                            pattern="*/train/*",
                            fs=44100,
                            seg_len=4,
                            overlap=2,
                            stft=True,
                            win_len=2048, 
                            hop_len=512, 
                            n_fft=2048,
                            complex_as_channels=True,
                            label=None)

    print(train_data[0])

if __name__ == '__main__':
    main()