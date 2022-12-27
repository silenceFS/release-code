import json
import os
import random

import numpy as np
import pandas as pd
import pickle as pkl

from base.base_dataset import TextVideoDataset_full


class MSVD(TextVideoDataset_full):
    def _load_metadata(self):
        caption_path = '/data/fs/dataset_video/msvd/raw-captions.pkl'
        caption_data = pkl.load(open(caption_path,'rb'))
        if self.split == 'train':
            path = '/data/fs/dataset_video/msvd/train_list.txt'
        elif self.split == 'val':
            path = '/data/fs/dataset_video/msvd/val_list.txt'
        else:
            path = '/data/fs/dataset_video/msvd/test_list.txt'
        video_list = open(path,'r').readlines()
        video_list = [video.strip() for video in video_list]
        self.metadata = {}
        for video in video_list:
            caption_list = caption_data[video]
            caption_list = [[' '.join(caption) for caption in caption_list]]
            self.metadata[video] = caption_list
        self.metadata = pd.DataFrame.from_dict(self.metadata, orient='index', columns=['captions'])

    def _get_video_path(self, sample):
        return os.path.join('/data/fs/dataset_video/msvd/YouTubeClips', sample.name + '.avi'), sample.name + '.avi'

    def _get_caption(self, sample):
        caption_sample = self.text_params.get('caption_sample', "rand")
        if self.split in ['train', 'val'] and caption_sample == "rand":
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption

    def _get_all_caption(self, sample):
        # captions = sample['captions']
        return sample['captions']