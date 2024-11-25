# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

class VideoData(Dataset):
    def __init__(self, mode: str, name: str, split_index: int):
        self.mode = mode # train or test
        self.name = name # summe or tvsum

        # Datasetlerin pathleri belirlenir. Dataset modeline göre ilgili datasetin pathi filename olarak atanır
        # Ardından ilgili datapath'teki h5 dosyası 'video_data' fieldına yükleniyor.
        self.datasets = ['./data/summe/eccv16_dataset_summe_google_pool5.h5','./data/tvsum/eccv16_dataset_tvsum_google_pool5.h5']
        if self.name == "summe":
            self.filename = self.datasets[0]
        elif self.name == "tvsum":
            self.filename = self.datasets[1]
        self.video_data = h5py.File(self.filename, 'r')


        self.splits_filename = f'./data/splits/{self.name}_splits.json'
        self.splits = [] # [{train_keys: [v1,v2,v3] test_keys: [v5,v6,v7]},{train_keys: [v1,v2,v3] test_keys: [v5,v6,v7]}]
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        # Splits dosyasından train_keys ve test_keys altındaki videoları aynı şekilde
        # objenin splits fieldına atar.
        with open(self.splits_filename) as f:
            data = json.loads(f.read())
            for d in data:
                split = {
                    'train_keys': d['train_keys'],
                    'test_keys': d['test_keys']
                }
                self.splits.append(split)

    def __len__(self):
        self.len = len(self.splits[0][self.mode+'_keys'])
        return self.len
    
    # In "train" mode it returns the features; in "test" mode it returns also the video_name
    def __getitem__(self, index):
        # Split_index seçilen split
        # self.mode_keys -> train_keys or test_keys
        # index ile video_1 gibi videolar alınıyor
        video_name = self.splits[self.split_index][self.mode + '_keys'][index]
        #print('VIDEO',video_name)
        #print(self.video_data['video_1/features'])
        frame_features = torch.Tensor(np.array(self.video_data[f"{video_name}/features"]))
        print('FRAME FEAT',len(frame_features))
        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features


def get_loader(mode: str,name: str, split_index: int):
    vd = VideoData(mode, name, split_index)

    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1) # TODO: Why no shuffle!
    else:
        return vd


if __name__ == '__main__':
    pass
