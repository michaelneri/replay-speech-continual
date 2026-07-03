import os
import torch
import numpy as np
import torchaudio, torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from lightning import LightningDataModule
import random
import torchvision.transforms.v2
import io
import zipfile


########## CONSTANTS
# map from protocal device id to device name
MIC = {1: 'AIY', 2: 'RES_4', 3: 'RES_CORE', 4: 'AMLOGIC'}

# the channel id used when n channel is tested for each device
MIC_ARRAY_CHANNEL = {'AIY': {1: [0], 2: [0, 1]},
                    'RES_4': {1: [0], 2: [0, 3], 3: [0, 1, 3], 4: [0, 1, 2, 3]},
                    'RES_CORE': {1: [0], 2: [0,3], 3: [0,1,3], 4: [0,1,3,4], 5: [0,1,2,3,4], 6: [0,1,2,3,4,5]},
                    'AMLOGIC': {1: [0], 2: [0,3], 3: [0,1,3], 4: [0,1,3,4], 5: [0,1,2,3,4], 6: [0,1,2,3,4,5], 7: [0,1,2,3,4,5,6]}} # the center mic is added at last

NUM_CHANNELS_MIC = {'AIY': 2, 'RES_4': 4, 'RES_CORE': 6, 'AMLOGIC': 7}
##########

# define the channel number kept
class AdjustChannel(torch.nn.Module):
    def __init__(self, channel_num):
        super(AdjustChannel, self).__init__()
        self.channel_num = channel_num

    def forward(self, sample):
        waveform = sample['waveform']
        device = sample['device']
        if waveform.size()[0] >= self.channel_num:
            use_channel = MIC_ARRAY_CHANNEL[MIC[device]][self.channel_num]
            waveform = waveform[use_channel, :]
            sample['waveform'] = waveform
        else:
            raise Exception('cannot get channel more than original channel numbers')
        return sample
    
class ChannelSelector(torch.nn.Module):
    def __init__(self, channel_list):
        super(ChannelSelector, self).__init__()
        if type(channel_list) == list:
            self.channel_list = channel_list
        else:
            raise Exception("channel_list muste be a list, found {}".format(type(channel_list)))

    def forward(self, sample):
        waveform = sample['waveform']
        device = sample['device']
        if waveform.size()[0] < len(self.channel_list):
            raise Exception('cannot get channel more than original channel numbers')
        else:
            waveform = waveform[self.channel_list, :]
            sample['waveform'] = waveform
        return sample

# cut or pad to a specific audio length (in number of data points)
class AdjustLength(torch.nn.Module):
    def __init__(self, audio_length):
        super(AdjustLength, self).__init__()
        self.audio_length = audio_length

    def forward(self, sample):
        waveform = sample['waveform']
        if waveform.size()[1] > self.audio_length:
            waveform = waveform[:, 0: self.audio_length]
        else:
            pad_len = self.audio_length - waveform.size()[1]
            pad_op = torch.nn.ZeroPad2d([0, pad_len, 0, 0])
            waveform = pad_op(waveform)
        sample['waveform'] = waveform
        return sample

# normalize the signal scale that has the max amplitude of 1 (-1), i.e., signal = 1 / max(signal) * signal
class NormScale(torch.nn.Module):
    def __init__(self, scale=1):
        super(NormScale, self).__init__()
        self.scale = scale

    def forward(self, sample):
        waveform = sample['waveform']
        max_scale = max(abs(waveform.max()), abs(waveform.min()))
        waveform = (1 / max_scale) * waveform * self.scale
        sample['waveform'] = waveform
        return sample
    

class Remasc(Dataset):
    def __init__(self, data_path, env='all', device='all', transform=AdjustLength(16000), percentage_to_use = 1):
        self.data_path = data_path
        self.zip_path = os.path.join(".", "Remasc_baseline_backup.zip")
        self.internal_path = "Remasc_baseline_backup/data/"+self.data_path
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
        with self.zip_file.open(self.internal_path + "/meta.csv") as f:
            self.meta = pd.read_csv(f, delimiter=',', dtype=str)
        self.meta.columns = self.meta.columns.str.strip()
        self.meta = self.meta.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.transform = transform
        self.percentage_to_use = percentage_to_use

        # only keep replayed (3) and genuine recording (2)
        self.meta = self.meta[self.meta.iloc[:, 1].astype(int) > 1]

        # only keep selected device, input should be a list, element value range from [1,2,3,4]
        if device != 'all':
            if isinstance(device, int):
                device = [device]
            self.meta = self.meta[self.meta.iloc[:, 7].astype(int).isin(device)]

        # only keep selected environment, input should be a list, element value range from [1,2,3,4]
        if env != 'all':
            if isinstance(env, int):
                env = [env]
            self.meta = self.meta[self.meta.iloc[:, 3].astype(int).isin(env)]

        self.meta_list = self.meta.values.tolist()
        print('Successfully load {} files.'.format(len(self.meta_list)))

        if self.percentage_to_use != 1:
            random.seed(6) # <--------------- for comparison between different runs
            self.meta_list = random.sample(self.meta_list, int(len(self.meta_list)*self.percentage_to_use))
            print("Only available files {}".format(len(self.meta_list)))



    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        item_meta = self.meta_list[idx]
        audio_name = item_meta[0] + '.wav'
        with self.zip_file.open(self.internal_path+"/data/"+audio_name) as audio_file:
            audio_bytes = io.BytesIO(audio_file.read())
            waveform, sample_rate = torchaudio.load(audio_bytes)

        audio_label = int(item_meta[1]) - 2
        device = int(item_meta[7])
        environment = int(item_meta[3])
        talker = int(item_meta[2])
        loudspeaker = int(item_meta[6])
        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'label': audio_label, 'device': device, 
                  'environment': environment, 'filename': audio_name, 'talker': talker, 'loudspeaker': loudspeaker}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RemascDataModule(LightningDataModule):
    def __init__(self, batch_size, env, rdevice, num_workers=0, transform=None, percentage_to_use=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.env = env
        self.rdevice = rdevice
        self.percentage_to_use = percentage_to_use
        self.train_dataset = Remasc("core", self.env, self.rdevice, self.transform, self.percentage_to_use)
        self.test_dataset = Remasc("eval", self.env, self.rdevice, self.transform)

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def count_genuine_and_replay(self):
        labels = self.train_dataset.meta.iloc[:, 1].astype(int)
        genuine_count = (labels == 2).sum()  #  2 is the label for genuine
        replay_count = (labels == 3).sum()   #  3 is the label for replay
        return genuine_count, replay_count

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)


# sample usage
if __name__ == '__main__':
    # Example 1: load a single file 
    # load a set, note: order of the transform does matter
    remasc_complete = Remasc("core", env=[1,2,3,4], device=[1], 
                             transform=torchvision.transforms.v2.Compose([AdjustChannel(2), AdjustLength(44100), NormScale()]))
    # randomly select an sample from the set
    sample = remasc_complete[np.random.randint(0, len(remasc_complete))]
    # get the waveform
    sample_wav = sample['waveform']
    print("Dimension of a single sample: {}".format(sample_wav.shape))
    # get the label
    label = sample['label']
    print("Label: {}".format(label))
    device = sample['device']
    print("Device: {}".format(device))
    environment = sample['environment']
    print("Environment: {}".format(environment))
    batch_size = 64
    try_datamodule = RemascDataModule(batch_size, [2], [1], 0, transform=torchvision.transforms.v2.Compose([AdjustChannel(2), AdjustLength(44100), NormScale()]))
    print("Len train dataloader {}".format(len(try_datamodule.train_dataloader())))
    print("Len val dataloder {}".format(len(try_datamodule.val_dataloader())))
    print("Len test dataloder {}".format(len(try_datamodule.test_dataloader())))
    print(try_datamodule.count_genuine_and_replay())



    ####
    print("TESTING DATAMODULE FOR SELECTING CHANNELS")
    training_device = 3
    training_microphone_list = [0]
    testing_device = 3
    testing_microphone_list = [1]
    environment = 3 # fixed environment since talkers, loudspeakers, and microphones (both genuine and spoofing) are variable

    try_datamodule_source = RemascDataModule(batch_size, env = [environment], rdevice = [training_device], num_workers=0, transform=torchvision.transforms.v2.Compose([ChannelSelector(training_microphone_list), AdjustLength(44100)]))
    try_datamodule_target = RemascDataModule(batch_size, env = [environment], rdevice = [testing_device], num_workers=0, transform=torchvision.transforms.v2.Compose([ChannelSelector(testing_microphone_list), AdjustLength(44100)]))

    print("Len train dataloader {}".format(len(try_datamodule_source.train_dataloader())))
    print("Len val dataloder {}".format(len(try_datamodule_target.val_dataloader())))
    print("Len test dataloder {}".format(len(try_datamodule_target.test_dataloader())))
    print("Training data")
    training_batch = next(iter(try_datamodule_source.train_dataloader()))
    print(training_batch['waveform'].shape)
    print(training_batch['label'])
    print(training_batch['environment'])
    print(training_batch['device'])
    print(try_datamodule_source.count_genuine_and_replay())
    print("Val/Test data")
    testing_batch = next(iter(try_datamodule_target.test_dataloader()))
    print(testing_batch['waveform'].shape)
    print(testing_batch['label'])
    print(testing_batch['environment'])
    print(testing_batch['device'])
