import torch
from torch.utils.data import Dataset
import torch.nn as nn

from torchvision.transforms import Normalize

import torchaudio
import numpy as np
import pandas as pd

#constants for augmentation
MAX_NOISE_FACTOR = 0.03
MAX_SHIFT = 0.5
MAX_PITCH_SHIFT_STEP = 6
#constants for normalization, calculated on the whole dataset
DATASET_KALDI_MEAN = -6.773274898529053
DATASET_KALDI_STD = 3.796977996826172
DATASET_RAW_MEAN = -0.00014325266238301992
DATASET_RAW_STD = 0.12926168739795685
DATASET_MEAN = -51.28519821166992
DATASET_STD = 46.351680755615234


class ESCdataset(Dataset):
    def __init__(self, path, folds=None, n_fft=1024, hop_length=512, n_mels=128,
                augment=True, pitch_shift=False, normalize=True, log_mel=True,
                use_kaldi=True, target_len=None, resample_rate=None) -> None:
        """
        @brief  ESC-50 dataset class
        @param[in]  path        path to the ESC-50 dataset folder
        @param[in]  folds       folds out of 5 possible
        @param[in]  n_fft, hop_length, n_mels parameters for logarithmic Mel spectogram
        @param[in]  augment     bool, should data augmentation be used
        @param[in]  pitch_shift bool, should pitch shift augmentation be used
        @param[in]  normalize   bool, should normalization be used
        @param[in]  log_mel     bool, should logmel transformation be used
        @param[in]  use_kaldi   bool, should kaldi.fbank be used for logmel
        @param[in]  target_len  int, target len for padding
        @param[in]  resample_rate   new sample rate
        """
        super().__init__()
        self.path = path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.augment = augment
        self.pitch_shift = pitch_shift
        self.normalize = normalize
        self.log_mel = log_mel
        self.use_kaldi = use_kaldi
        self.target_len = target_len

        #reading the csv file for creating labels
        alldata = pd.read_csv(path + "/meta/esc50.csv").to_numpy()
        #filename and the index/code of the category
        self.csv_data = alldata[:, [0, 2]]
        #selecting the required folds
        if folds is not None:
            fold_index = np.isin(alldata[:, 1], folds)
            self.csv_data = self.csv_data[fold_index]
        #dictionary, then list for the indexes and category names
        label_dict = dict(alldata[:, [2, 3]])
        self.label_list = [i for _, i in sorted(label_dict.items())]

        #all files should have the same sample rate, saving based on the first file
        _, original_sr = torchaudio.load(
            self.path + "/audio/" + self.csv_data[0, 0])
        self.sample_rate = original_sr if resample_rate is None else resample_rate

        #instantiating the classes for transfoming the data
        self.Resample = self._create_resample(original_sr, self.sample_rate)
        self.LogMelSpect = self._create_log_mel_spect()
        self.Normalize = self._create_normalize()

    def __len__(self) -> int:
        """@returns the length of the dataset"""
        return len(self.csv_data)

    def __getitem__(self, item: int):
        """@returns (data, label)        
        @param[in]  item    index in the csv file
        """
        data = self._get_data(item)
        label = self._get_label(item)
        return data, label

    def _get_label(self, item: int):
        """@returns label based on the index"""
        return self.csv_data[item, 1]
    
    def get_class_name(self, index:int):
        """@returns class name based on the index"""
        return self.label_list[index]

    def _get_data(self, item: int):
        """@returns the transformed data based on the index
        @param[in]  item    index in the csv file
        """
        data = self.get_waveform(item)
        data = self.Resample(data)
        if self.augment:
            data = self.transform(data, MAX_SHIFT, MAX_PITCH_SHIFT_STEP,
                                MAX_NOISE_FACTOR, self.pitch_shift)
        if self.log_mel:
            if self.LogMelSpect is None:
                data = torchaudio.compliance.kaldi.fbank(
                    data, htk_compat=True, sample_frequency=self.sample_rate,
                    use_energy=False, window_type='hanning',
                    num_mel_bins=self.n_mels, dither=0.0,
                    frame_shift=self.frame_shift,frame_length=self.frame_length
                )
                data = data.unsqueeze(0)
            else:
                data = self.LogMelSpect(data)
        if self.normalize and self.log_mel:
            data = self.Normalize(data)
        elif self.normalize:
            data = (data - DATASET_RAW_MEAN) / DATASET_RAW_STD
        #padding to target_len
        if self.target_len is not None and self.log_mel and self.use_kaldi:
            #source: AST transformer git repo
            n_frames = data.shape[1]
            p = self.target_len - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                data = m(data)
            elif p < 0:
                data = data[0:self.target_len, :]

        return torch.squeeze(data)

    def get_waveform(self, item: int):
        """@returns the raw waveform data based on the index
        @param[in]  item    index in the csv file
        """
        data, _ = torchaudio.load(self.path + "/audio/" + self.csv_data[item, 0])
        return data

    def _create_log_mel_spect(self):
        """@returns an instance of nn.Sequential(transforms.MelSpectogram,
                                                transforms.AmplitudeToDB)
        """
        to_melspect = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude")

        to_log_mel_spect = nn.Sequential(to_melspect, to_db)
        if self.use_kaldi:
            self.frame_shift=self.hop_length*1000/self.sample_rate
            self.frame_length=self.n_fft*1000/self.sample_rate
        return to_log_mel_spect if not self.use_kaldi else None

    def _create_resample(self, original_sr, new_sr):
        """@returns an instance of transforms.Resample"""
        return torchaudio.transforms.Resample(
            original_sr, new_sr)

    #TODO: classes for augmentation and calling them with help of nn.Sequential

    def inject_noise(self, data, max_noise_factor):
        """@brief   adds white noise with random standard deviation to the data
        @returns    the augmented data
        @param[in]  max_noise_factor    max of the standard deviation
        """
        #random muliplier between 0 and max_noise_factor
        noise_factor = torch.rand(1) * max_noise_factor
        #white noise
        noise = torch.randn(data.size(1))
        augmented_data = data + noise_factor * noise
        return augmented_data

    def pitch_shift_rand(self, data, max_step):
        """@brief   transforms the data using random pitch shift        
        @returns    the augmented data
        @param[in]  max_step    max of the n_steps parameter for functional.pitch_shift
        """
        #random n_steps
        n_steps = np.random.randint(-max_step, max_step + 1)
        augmented_data = torchaudio.functional.pitch_shift(
            data, self.sample_rate, n_steps)
        return augmented_data

    def time_shift_rand(self, data, shift_max):
        """@returns the augmented data
        @param[in]  shift_max   max shift in seconds
        """
        #conversion from seconds
        shift_max_steps = int(shift_max * self.sample_rate)
        shift = np.random.randint(-shift_max_steps, shift_max_steps)
        #shifting the data
        augmented_data = torch.roll(data, shift)
        if shift >= 0:
            augmented_data[0, :shift] = 0
        else:
            augmented_data[0, shift:] = 0
        return augmented_data

    def standardize(self, data):
        """@brief   dividing the data with the maximum value"""
        max = torch.max(data)
        return data/max

    def _create_normalize(self):
        """
        @brief  returns an instance of torchvision.transforms.Normalize
                with the previously calculated mean and std 
        """
        return Normalize(DATASET_KALDI_MEAN if self.use_kaldi else DATASET_MEAN,
                        2 * DATASET_KALDI_STD if self.use_kaldi else DATASET_STD)

    def transform(self, data, shift_max, max_step,
                    max_noise_factor, pitch_shift=False):
        """
        @brief      applies time shift, pitch shift, noise injection
        @returns    the augmented data
        """
        #data = standardize(data)
        data = self.time_shift_rand(data, shift_max)
        if pitch_shift:
            data = self.pitch_shift_rand(data, max_step)
        data = self.inject_noise(data, max_noise_factor)
        return data