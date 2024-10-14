import torch
import torchaudio
import os

# a template of dataset for building reference_dataset and generated_dataset
class dataset_template(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
        self.datalist = sorted(self.datalist)

    def __getitem__(self, index):
        filename = self.datalist[index]
        waveform = self.read_from_file(filename)
        return waveform, os.path.basename(filename)

    def __len__(self):
        return len(self.datalist)
    
    def read_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # only use the first channel    
        audio = audio.squeeze(0)
        # min-max normalization
        return audio