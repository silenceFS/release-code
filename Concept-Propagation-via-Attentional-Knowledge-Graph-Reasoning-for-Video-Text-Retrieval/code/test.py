import argparse

import pandas as pd
import torch
import transformers
from sacred import Experiment
from tqdm import tqdm

import data_loader.data_loader as module_data
import model.metric as module_metric
import model.model as module_arch
from model.model import sim_matrix
from parse_config import ConfigParser
from trainer.trainer import verbose
from utils.util import state_dict_data_parallel_fix
import numpy as np
import os

from trainer import cal_perf

ex = Experiment('test')


@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    data_loader = config.initialize('data_loader', module_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    text_embed_arr_semantic = []
    vid_embed_arr_semantic = []
    print(len(data_loader))

    if config['data_loader']['args']['dataset_name'] == 'MSRVTT-full':
        with torch.no_grad():
            v2t_gt = []
            count = 0
            for i, data in tqdm(tqdm(enumerate(data_loader))):
                # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])
                all_text = []
                for ii in range(data['video'].shape[0]):
                    v2t_gt.append([])
                    for jj in range(20):
                        all_text.append(data['text'][jj][ii])
                        v2t_gt[-1].append(count)
                        count += 1
                data['text'] = all_text
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                if isinstance(data['video'], list):
                    data['video'] = [x.to(device) for x in data['video']]
                else:
                    data['video'] = data['video'].to(device)

                text_embed, vid_embed, text_embed_semantic, vid_embed_semantic = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())
                text_embed_arr_semantic.append(text_embed_semantic.cpu().detach())
                vid_embed_arr_semantic.append(vid_embed_semantic.cpu().detach())

        t2v_gt = {}
        for i, t_gts in enumerate(v2t_gt):
            for t_gt in t_gts:
                t2v_gt.setdefault(t_gt, [])
                t2v_gt[t_gt].append(i)

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)
        text_embeds_semantic = torch.cat(text_embed_arr_semantic)
        vid_embeds_semantic = torch.cat(vid_embed_arr_semantic)
    
    elif config['data_loader']['args']['dataset_name'] == 'MSVD':

        with torch.no_grad():
            v2t_gt = []
            count = 0
            for i, data in tqdm(tqdm(enumerate(data_loader))):
                # leave this for now since not doing anything on the gpu
                for n in data['meta']:
                    v2t_gt.append([])
                    for ii in range(n):
                        v2t_gt[-1].append(count)
                        count += 1
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                # if isinstance(data['video'], list):
                #     data['video'] = [x.to(device) for x in data['video']]
                # else:
                data['video'] = data['video'].to(device)

                text_embed, vid_embed, text_embed_semantic, vid_embed_semantic = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())
                text_embed_arr_semantic.append(text_embed_semantic.cpu().detach())
                vid_embed_arr_semantic.append(vid_embed_semantic.cpu().detach())
        
        t2v_gt = {}
        for i, t_gts in enumerate(v2t_gt):
            for t_gt in t_gts:
                t2v_gt.setdefault(t_gt, [])
                t2v_gt[t_gt].append(i)

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)
        text_embeds_semantic = torch.cat(text_embed_arr_semantic)
        vid_embeds_semantic = torch.cat(vid_embed_arr_semantic)

    else:
        with torch.no_grad():
            for i, data in tqdm(tqdm(enumerate(data_loader))):
                # leave this for now since not doing anything on the gpu
                meta_arr.append(data['meta'])
                if tokenizer is not None:
                    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                data['text'] = {key: val.cuda() for key, val in data['text'].items()}
                if isinstance(data['video'], list):
                    data['video'] = [x.to(device) for x in data['video']]
                else:
                    data['video'] = data['video'].to(device)

                text_embed, vid_embed, text_embed_semantic, vid_embed_semantic = model(data, return_embeds=True)
                text_embed_arr.append(text_embed.cpu().detach())
                vid_embed_arr.append(vid_embed.cpu().detach())
                text_embed_arr_semantic.append(text_embed_semantic.cpu().detach())
                vid_embed_arr_semantic.append(vid_embed_semantic.cpu().detach())

        text_embeds = torch.cat(text_embed_arr)
        vid_embeds = torch.cat(vid_embed_arr)
        text_embeds_semantic = torch.cat(text_embed_arr_semantic)
        vid_embeds_semantic = torch.cat(vid_embed_arr_semantic)

    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        cpu_vid_embeds = vid_embeds
        cpu_text_embeds = text_embeds
        cpu_vid_embeds_semantic = vid_embeds_semantic
        cpu_text_embeds_semantic = text_embeds_semantic

        li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        li_vid_embeds_semantic = [x for x in cpu_vid_embeds_semantic]
        li_txt_embeds_semantic = [x for x in cpu_text_embeds_semantic]
        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
        vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                               'vid_embed_semantic': li_vid_embeds_semantic, 'txt_embed_semantic': li_txt_embeds_semantic,
                               'captions': raw_caps})
        new_vid_embeds = []
        new_txt_embeds = []
        new_vid_embeds_semantic = []
        new_txt_embeds_semantic = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)
            tvembeds_semantic = torch.stack(tdf['vid_embed_semantic'].values.tolist())
            tvembeds_semantic = tvembeds_semantic.mean(dim=0)
            new_vid_embeds_semantic.append(tvembeds_semantic)

            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])
                ttembeds_semantic = torch.stack(cdf['txt_embed_semantic'].values.tolist())
                new_txt_embeds_semantic.append(ttembeds_semantic[0])

        vid_embeds = torch.stack(new_vid_embeds)
        text_embeds = torch.stack(new_txt_embeds)
        vid_embeds_semantic = torch.stack(new_vid_embeds_semantic)
        text_embeds_semantic = torch.stack(new_txt_embeds_semantic)

    if args.split != 'train':  # because train is usually too big
        # text_embeds = torch.cat((text_embeds,text_embeds_semantic),1)
        # vid_embeds = torch.cat((vid_embeds,vid_embeds_semantic),1)
        # sims = sim_matrix(text_embeds, vid_embeds)
        sims = 0.6 * sim_matrix(text_embeds, vid_embeds) + 0.4 * sim_matrix(text_embeds_semantic, vid_embeds_semantic)
        sims = sims.numpy()

        if config['data_loader']['args']['dataset_name'] == 'MSRVTT-full' or config['data_loader']['args']['dataset_name'] == 'MSVD':
            t2v_all_errors = -1 * sims
            (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score), (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score) = cal_perf(t2v_all_errors, v2t_gt, t2v_gt)
        else:
            nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims, query_masks=mask)
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                nested_metrics[metric_name] = res

    # if config.config['visualizer']:
    #    raise NotImplementedError
    if args.save_feats is not None:
        vid_embeds = vid_embeds.cpu().detach().numpy()
        text_embeds = text_embeds.cpu().detach().numpy()
        vid_embeds_save_fp = os.path.join(args.save_feats, f'vid_embeds_{data_loader.dataset.split}.npy')
        txt_embeds_save_fp = os.path.join(args.save_feats, f'txt_embeds_{data_loader.dataset.split}.npy')

        np.save(vid_embeds_save_fp, vid_embeds)
        np.save(txt_embeds_save_fp, text_embeds)

        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        videoids.to_csv(os.path.join(args.save_feats, f'ids_{data_loader.dataset.split}.csv'), index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=16, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True)
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
