import json
import os
import random

import numpy as np
import pandas as pd

from base.base_dataset import TextVideoDataset


class ActivityNet(TextVideoDataset):
    def _load_metadata(self):
        if self.split == 'train':
            caption_path = os.path.join(self.metadata_dir, 'captions', 'frozen-in-time', 'train_delete.json')
        else:
            caption_path = os.path.join(self.metadata_dir, 'captions', 'frozen-in-time', 'val_add_2.json')
        self.metadata = json.load(open(caption_path,'r'))
        self.metadata = pd.DataFrame.from_dict(self.metadata, orient='index', columns=['captions'])

        return self.metadata

    def _get_video_path(self, sample):
        return os.path.join(self.data_dir, 'raw_videos', sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        return sample['captions']